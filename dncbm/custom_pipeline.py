from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from jaxtyping import Int64
from pydantic import NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult


from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)

from time import time


from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.tensor_types import Axis


if TYPE_CHECKING:
    from sparse_autoencoder.metrics.abstract_metric import MetricResult


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    optimizer: AbstractOptimizerWithReset
    """Optimizer to use."""

    progress_bar: tqdm | None
    """Progress bar for the pipeline."""

    total_activations_trained_on: int = 0
    """Total number of activations trained on state."""

    @property
    def n_components(self) -> int:
        """Number of source model components the SAE is trained on."""

        return 1  # since we are training on a single component, which is the out layer

    def __init__(
        self,
        activation_resampler: AbstractActivationResampler | None,
        autoencoder: SparseAutoencoder,
        loss: AbstractLoss,
        optimizer: AbstractOptimizerWithReset,
        checkpoint_directory: Path = None,
        log_frequency: PositiveInt = 100,
        metrics: MetricsContainer = default_metrics,
        device: torch.cuda = 'cuda',
        args=None
    ) -> None:

        self.activation_resampler = activation_resampler
        self.autoencoder = autoencoder
        self.checkpoint_directory = checkpoint_directory
        self.log_frequency = log_frequency
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.args = args

    @validate_call(config={"arbitrary_types_allowed": True})
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: PositiveInt
    ) -> tuple[Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)], dict]:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Tuple of (learned_activations_fired_count, metrics_dict) where metrics_dict contains:
            - total_loss: Average total loss
            - reconstruction_loss: Average reconstruction (L2) loss
            - l1_loss: Average L1 sparsity loss
            - l0_sparsity: Average number of active features
            - n_batches: Number of batches processed
        """

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
            shuffle=True
        )

        learned_activations_fired_count: Int64[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.zeros(
            (self.n_components, self.autoencoder.n_learned_features),
            dtype=torch.int64,
            device=self.device,)

        # Accumulate metrics
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        l1_loss_sum = 0.0
        l0_sum = 0.0
        n_batches = 0

        for id, store_batch in enumerate(activations_dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(self.device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder.forward(
                batch)

            # Get loss & metrics
            metrics: list[MetricResult] = []
            total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                batch,
                learned_activations,
                reconstructed_activations,
                component_reduction=LossReductionType.MEAN
            )
            metrics.extend(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations,
                                        reconstructed_activations)
                    )
                    metrics.extend(calculated)

            # Accumulate metrics for epoch summary (BEFORE backward pass for performance)
            with torch.no_grad():
                # Store count of how many neurons have fired
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))
                
                # Accumulate metrics for epoch summary
                total_loss_sum += total_loss.item()
                
                # Extract L1 and reconstruction losses from loss_metrics
                # Metrics have a 'postfix' field with the actual loss name
                found_recon = False
                found_l1 = False
                
                for loss_metric in loss_metrics:
                    postfix = (loss_metric.postfix or '').lower()
                    name = loss_metric.name.lower()
                    
                    # Handle tensor values
                    val = loss_metric.component_wise_values
                    if isinstance(val, torch.Tensor):
                        if val.numel() == 1:
                            val_scalar = val.item()
                        else:
                            val_scalar = float(val.mean().item())
                    else:
                        val_scalar = float(val)
                    
                    # Match by postfix (more specific) or name
                    if 'learned_activations_l1' in postfix or (postfix == '' and 'l1' in name):
                        l1_loss_sum += val_scalar
                        found_l1 = True
                    elif 'l2_reconstruction' in postfix or 'reconstruction' in postfix or (postfix == '' and 'reconstruction' in name):
                        recon_loss_sum += val_scalar
                        found_recon = True
                
                # Fallback: compute directly if not found in metrics (only if needed)
                if not found_recon:
                    # Compute reconstruction loss manually
                    if reconstructed_activations.dim() == 3:
                        recon_flat = reconstructed_activations.squeeze(1)
                        batch_flat = batch.squeeze(1)
                    else:
                        recon_flat = reconstructed_activations
                        batch_flat = batch
                    recon_loss_batch = torch.nn.functional.mse_loss(recon_flat, batch_flat, reduction='mean')
                    recon_loss_sum += recon_loss_batch.item()
                
                if not found_l1 and hasattr(self.args, 'l1_coeff'):
                    # Compute L1 loss manually
                    if learned_activations.dim() == 3:
                        l1_batch = learned_activations.abs().sum(dim=-1).mean() * self.args.l1_coeff
                    else:
                        l1_batch = learned_activations.abs().mean() * self.args.l1_coeff
                    l1_loss_sum += l1_batch.item()
                
                # Calculate L0 sparsity (average number of active features per sample)
                if learned_activations.dim() == 3:  # [batch, component, features]
                    l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
                else:  # [batch, features]
                    l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
                l0_sum += l0
                n_batches += 1

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.post_backwards_hook()

            # Log training metrics (ORIGINAL LOGGING - PRESERVE ALL METRICS)
            self.total_activations_trained_on += train_batch_size
            if (
                wandb.run is not None
                and int(self.total_activations_trained_on / train_batch_size) % self.log_frequency
                == 0
            ):
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                wandb.log(
                    log,
                    step=self.total_activations_trained_on,
                    commit=False,
                )
        
        # Compute averages
        metrics_dict = {
            'total_loss': total_loss_sum / n_batches if n_batches > 0 else 0.0,
            'reconstruction_loss': recon_loss_sum / n_batches if n_batches > 0 else 0.0,
            'l1_loss': l1_loss_sum / n_batches if n_batches > 0 else 0.0,
            'l0_sparsity': l0_sum / n_batches if n_batches > 0 else 0.0,
            'n_batches': n_batches
        }
        
        return learned_activations_fired_count, metrics_dict

    def save_checkpoint(self, *, is_final: bool = False) -> Path:
        """Save the model as a checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        # Create the name
        name: str = f"sparse_autoencoder_{'final' if is_final else self.total_activations_trained_on}"
        safe_name = quote_plus(name, safe="_")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        file_path: Path = self.checkpoint_directory / f"{safe_name}.pt"

        torch.save(
            self.autoencoder.state_dict(),
            file_path,
        )
        return file_path

    def update_parameters(self, parameter_updates: list[ParameterUpdateResults]) -> None:
        """Update the parameters of the model from the results of the resampler.

        Args:
            parameter_updates: Parameter updates from the resampler.
        """
        for component_idx, component_parameter_update in enumerate(parameter_updates):
            # Update the weights and biases
            self.autoencoder.encoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_weight_updates,
                component_idx=component_idx,
            )
            self.autoencoder.encoder.update_bias(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_bias_updates,
                component_idx=component_idx,
            )
            self.autoencoder.decoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_decoder_weight_updates,
                component_idx=component_idx,
            )

            # Reset the optimizer
            for parameter, axis in self.autoencoder.reset_optimizer_parameter_details:
                self.optimizer.reset_neurons_state(
                    parameter=parameter,
                    neuron_indices=component_parameter_update.dead_neuron_indices,
                    axis=axis,
                    component_idx=component_idx,
                )

    def get_activation_store(self, activation_fname):
        activations = torch.load(activation_fname)
        activation_store = TensorActivationStore(
            activations.shape[0], self.autoencoder.n_input_features, self.n_components)
        activation_store.empty()
        activation_store.extend(activations, component_idx=0)
        return activation_store

    # considering train_val_fnames to contain a single fname
    def validation(self, activation_store, train_batch_size):
        activations_dataloader = DataLoader(
            activation_store, batch_size=train_batch_size, shuffle=True)

        with torch.no_grad():
            total_loss_sum = 0.0
            recon_loss_sum = 0.0
            l1_loss_sum = 0.0
            l0_sum = 0.0
            r2_sum = 0.0
            n_batches = 0
            
            with tqdm(desc="Validation", total=len(activations_dataloader),) as progress_bar:
                for batch_id, store_batch in enumerate(activations_dataloader):
                    batch = store_batch.detach().to(self.device)
                    # Forward pass
                    learned_activations, reconstructed_activations = self.autoencoder.forward(
                        batch)
                    total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                        batch,
                        learned_activations,
                        reconstructed_activations,
                        component_reduction=LossReductionType.MEAN
                    )
                    
                    # Accumulate metrics
                    total_loss_sum += total_loss.item()
                    
                    # Extract L1 and reconstruction losses
                    # Metrics have a 'postfix' field with the actual loss name
                    for loss_metric in loss_metrics:
                        postfix = (loss_metric.postfix or '').lower()
                        name = loss_metric.name.lower()
                        
                        # Handle tensor values
                        val = loss_metric.component_wise_values
                        if isinstance(val, torch.Tensor):
                            if val.numel() == 1:
                                val_scalar = val.item()
                            else:
                                val_scalar = float(val.mean().item())
                        else:
                            val_scalar = float(val)
                        
                        # Match by postfix (more specific) or name
                        if 'learned_activations_l1' in postfix or (postfix == '' and 'l1' in name):
                            l1_loss_sum += val_scalar
                        elif 'l2_reconstruction' in postfix or 'reconstruction' in postfix or (postfix == '' and 'reconstruction' in name):
                            recon_loss_sum += val_scalar
                    
                    # Calculate L0 sparsity
                    if learned_activations.dim() == 3:
                        l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
                    else:
                        l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
                    l0_sum += l0
                    
                    # Calculate R² score
                    if reconstructed_activations.dim() == 3:
                        recon_flat = reconstructed_activations.squeeze(1)
                        batch_flat = batch.squeeze(1)
                    else:
                        recon_flat = reconstructed_activations
                        batch_flat = batch
                    
                    ss_res = ((batch_flat - recon_flat) ** 2).sum()
                    ss_tot = ((batch_flat - batch_flat.mean(dim=0, keepdim=True)) ** 2).sum()
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    r2_sum += r2.item()
                    
                    n_batches += 1
                    progress_bar.update(1)
            
            metrics_dict = {
                'total_loss': total_loss_sum / n_batches if n_batches > 0 else 0.0,
                'reconstruction_loss': recon_loss_sum / n_batches if n_batches > 0 else 0.0,
                'l1_loss': l1_loss_sum / n_batches if n_batches > 0 else 0.0,
                'l0_sparsity': l0_sum / n_batches if n_batches > 0 else 0.0,
                'r2_score': r2_sum / n_batches if n_batches > 0 else 0.0,
                'n_batches': n_batches
            }
            
            # Keep loss_metrics for wandb logging compatibility
            total_losses = torch.zeros((4, len(activations_dataloader)))
            # Return both for backward compatibility
            return loss_metrics, metrics_dict

    def run_pipeline(
        self,
        train_batch_size: PositiveInt,
        val_frequency: NonNegativeInt | None = None,
        checkpoint_frequency: NonNegativeInt | None = None,
        num_epochs=None,
        train_fnames=None,
        train_val_fnames=None,
        start_time=0,
        resample_epoch_freq: NonNegativeInt = 0,
    ) -> None:

        last_checkpoint: int = 0

        assert (train_fnames is not None)
        num_train_pieces = len(train_fnames)
        train_order = torch.randperm(num_train_pieces)
        train_piece_idx = 0

        self.actual_epochs = num_epochs * num_train_pieces
        pieces_per_epoch = num_train_pieces
        
        print(f"\n{'='*80}")
        print(f"Starting SAE Training")
        print(f"{'='*80}")
        print(f"Total epochs: {num_epochs}")
        print(f"Pieces per epoch: {pieces_per_epoch} (one piece = one training file)")
        print(f"Total iterations: {self.actual_epochs} (epochs × pieces)")
        print(f"{'='*80}")
        print(f"\nNOTE: One 'epoch' processes all {pieces_per_epoch} training files.")
        print(f"      Progress bar shows total iterations: {self.actual_epochs}\n")

        with tqdm(
            desc="Training Progress",
            total=self.actual_epochs,
        ) as progress_bar:

            self.progress_bar = progress_bar
            best_val_loss = float('inf')
            
            # Track metrics across pieces for epoch-level summaries
            epoch_train_metrics = []
            epoch_val_metrics = []

            for epoch in range(self.actual_epochs):
                # Calculate logical epoch number (0-based)
                logical_epoch = epoch // pieces_per_epoch
                piece_in_epoch = epoch % pieces_per_epoch
                
                # if the train activations are saved in more than one piece, shuffle the order
                if train_piece_idx >= num_train_pieces:
                    train_order = torch.randperm(num_train_pieces)
                    train_piece_idx = 0

                train_activation_store = self.get_activation_store(
                    train_fnames[train_order[train_piece_idx]])
                
                if piece_in_epoch == 0:
                    print(f"\n{'─'*80}")
                    print(f"Epoch {logical_epoch + 1}/{num_epochs} | Piece {piece_in_epoch + 1}/{pieces_per_epoch}")
                    print(f"{'─'*80}")
                
                train_piece_idx += 1
                self.current_epoch = epoch

                # Update the counters
                n_activation_vectors_in_store = len(train_activation_store)
                last_checkpoint += n_activation_vectors_in_store

                # Train
                progress_bar.set_postfix({"stage": f"train (ep {logical_epoch+1})"})
                batch_neuron_activity, train_metrics = self.train_autoencoder(
                    train_activation_store, train_batch_size=train_batch_size
                )
                epoch_train_metrics.append(train_metrics)

                # Resample dead neurons (if needed)
                # Note: resample based on logical epochs, not pieces
                if (self.activation_resampler is not None) and (resample_epoch_freq > 0) and \
                   (piece_in_epoch == pieces_per_epoch - 1) and ((logical_epoch + 1) % resample_epoch_freq == 0) and \
                   (logical_epoch + 1 < num_epochs):
                    progress_bar.set_postfix({"stage": "resampling"})
                    parameter_updates = self.activation_resampler.step_resampler(
                        batch_neuron_activity=batch_neuron_activity,
                        activation_store=train_activation_store,
                        autoencoder=self.autoencoder,
                        loss_fn=self.loss,
                        train_batch_size=train_batch_size,
                    )

                    if parameter_updates is not None:
                        total_dead = sum(len(update.dead_neuron_indices) for update in parameter_updates)
                        if total_dead > 0:
                            print(f"  → Resampling: {total_dead} dead neurons")
                            if wandb.run is not None:
                                wandb.log(
                                    {
                                        "resample/dead_neurons": [
                                            len(update.dead_neuron_indices)
                                            for update in parameter_updates
                                        ]
                                    },
                                    commit=False,
                                )
                            for id, update in enumerate(parameter_updates):
                                if len(update.dead_neuron_indices) > 0:
                                    print(f"    Component {id}: {len(update.dead_neuron_indices)} dead neurons")
                            # Update the parameters
                            self.update_parameters(parameter_updates)

                del train_activation_store

                # Get validation metrics (if needed) - only at end of epoch
                # val_frequency is now interpreted as "every N epochs"
                should_validate = (val_frequency > 0 and 
                                 piece_in_epoch == pieces_per_epoch - 1 and
                                 (logical_epoch + 1) % val_frequency == 0)
                
                if should_validate:
                    progress_bar.set_postfix({"stage": "validate"})
                    assert (train_val_fnames is not None)

                    val_metrics_list = []
                    all_val_loss_metrics = []  # Store all loss metrics for WandB
                    
                    for id, train_val_fname in enumerate(train_val_fnames):
                        train_val_activation_store = self.get_activation_store(
                            train_val_fname)
                        loss_metrics, val_metrics = self.validation(train_val_activation_store,
                                                                        train_batch_size)
                        val_metrics_list.append(val_metrics)
                        all_val_loss_metrics.append(loss_metrics)  # Keep original metrics
                        del train_val_activation_store
                    
                    # Average metrics across validation files
                    avg_val_metrics = {
                        'total_loss': sum(m['total_loss'] for m in val_metrics_list) / len(val_metrics_list),
                        'reconstruction_loss': sum(m['reconstruction_loss'] for m in val_metrics_list) / len(val_metrics_list),
                        'l1_loss': sum(m['l1_loss'] for m in val_metrics_list) / len(val_metrics_list),
                        'l0_sparsity': sum(m['l0_sparsity'] for m in val_metrics_list) / len(val_metrics_list),
                        'r2_score': sum(m['r2_score'] for m in val_metrics_list) / len(val_metrics_list),
                    }
                    epoch_val_metrics.append(avg_val_metrics)

                    # Check for best model
                    if avg_val_metrics['total_loss'] < best_val_loss:
                        best_val_loss = avg_val_metrics['total_loss']
                        print(f"  ✓ New best validation loss: {best_val_loss:.6f}")

                    # Log to WandB - PRESERVE ORIGINAL FORMAT
                    if wandb.run is not None:
                        log = {}
                        
                        # Log original validation metrics format (from loss_metrics)
                        # Average the loss metrics across validation files
                        if all_val_loss_metrics and len(all_val_loss_metrics) > 0:
                            # Get the first loss_metrics to see structure
                            first_loss_metrics = all_val_loss_metrics[0]
                            
                            # For each metric type, average across all validation files
                            for metric_idx, first_metric in enumerate(first_loss_metrics):
                                # Collect values from all validation files for this metric
                                metric_values = []
                                for lm_list in all_val_loss_metrics:
                                    if metric_idx < len(lm_list):
                                        lm = lm_list[metric_idx]
                                        val = lm.component_wise_values
                                        if isinstance(val, torch.Tensor):
                                            if val.numel() == 1:
                                                metric_values.append(val.item())
                                            else:
                                                metric_values.append(float(val.mean().item()))
                                        else:
                                            metric_values.append(float(val))
                                
                                if metric_values:
                                    # Average across validation files
                                    avg_value = sum(metric_values) / len(metric_values)
                                    
                                    # Update the metric and use its wandb_log property
                                    first_metric.location = MetricLocation.VALIDATE
                                    first_metric.component_wise_values = torch.tensor([avg_value])
                                    log.update(first_metric.wandb_log)
                        
                        # Also log our custom aggregated metrics for convenience
                        log.update({
                            'val/total_loss': avg_val_metrics['total_loss'],
                            'val/reconstruction_loss': avg_val_metrics['reconstruction_loss'],
                            'val/l1_loss': avg_val_metrics['l1_loss'],
                            'val/l0_sparsity': avg_val_metrics['l0_sparsity'],
                            'val/r2_score': avg_val_metrics['r2_score'],
                        })
                        
                        wandb.log(log, step=self.total_activations_trained_on, commit=True)
                    
                    # Print epoch summary immediately after validation
                    if epoch_train_metrics:
                        self._print_epoch_summary(logical_epoch, epoch_train_metrics, epoch_val_metrics)
                        # Clear metrics for next epoch (but keep last validation for reference)
                        epoch_train_metrics = []
                        epoch_val_metrics = []
                elif piece_in_epoch == pieces_per_epoch - 1:
                    # Print summary even if validation didn't run this epoch
                    if epoch_train_metrics:
                        self._print_epoch_summary(logical_epoch, epoch_train_metrics, [])
                        epoch_train_metrics = []
                        epoch_val_metrics = []

                # Checkpoint (if needed)
                if checkpoint_frequency != 0 and last_checkpoint >= checkpoint_frequency:
                    progress_bar.set_postfix({"stage": "checkpoint"})
                    last_checkpoint = 0
                    ckpt_path = self.save_checkpoint()
                    print(f"  → Checkpoint saved: {ckpt_path.name}")

                # Update the progress bar
                progress_bar.update(1)
            
            # Print final epoch summary if there are remaining metrics
            if epoch_train_metrics:
                self._print_epoch_summary(logical_epoch, epoch_train_metrics, epoch_val_metrics)

        # Save the final checkpoint
        final_path = self.save_checkpoint(is_final=True)
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Final checkpoint: {final_path}")
        print(f"Total time: {time() - start_time:.2f}s ({(time() - start_time)/60:.1f} minutes)")
        print(f"{'='*80}\n")
    
    def _print_epoch_summary(self, epoch: int, train_metrics_list: list, val_metrics_list: list):
        """Print a formatted summary of epoch metrics."""
        # Average training metrics across all pieces in epoch
        avg_train = {
            'total_loss': sum(m['total_loss'] for m in train_metrics_list) / len(train_metrics_list),
            'reconstruction_loss': sum(m['reconstruction_loss'] for m in train_metrics_list) / len(train_metrics_list),
            'l1_loss': sum(m['l1_loss'] for m in train_metrics_list) / len(train_metrics_list),
            'l0_sparsity': sum(m['l0_sparsity'] for m in train_metrics_list) / len(train_metrics_list),
        }
        
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch + 1} Summary")
        print(f"{'─'*80}")
        print(f"TRAIN:")
        print(f"  Total Loss:        {avg_train['total_loss']:.6f}")
        print(f"  Reconstruction:    {avg_train['reconstruction_loss']:.6f}")
        print(f"  L1 Sparsity:       {avg_train['l1_loss']:.6f}")
        print(f"  L0 Active Feat:    {avg_train['l0_sparsity']:.1f}")
        
        if val_metrics_list:
            avg_val = {
                'total_loss': sum(m['total_loss'] for m in val_metrics_list) / len(val_metrics_list),
                'reconstruction_loss': sum(m['reconstruction_loss'] for m in val_metrics_list) / len(val_metrics_list),
                'l1_loss': sum(m['l1_loss'] for m in val_metrics_list) / len(val_metrics_list),
                'l0_sparsity': sum(m['l0_sparsity'] for m in val_metrics_list) / len(val_metrics_list),
                'r2_score': sum(m['r2_score'] for m in val_metrics_list) / len(val_metrics_list),
            }
            print(f"\nVALIDATION:")
            print(f"  Total Loss:        {avg_val['total_loss']:.6f}")
            print(f"  Reconstruction:    {avg_val['reconstruction_loss']:.6f}")
            print(f"  L1 Sparsity:       {avg_val['l1_loss']:.6f}")
            print(f"  L0 Active Feat:    {avg_val['l0_sparsity']:.1f}")
            print(f"  R² Score:          {avg_val['r2_score']:.4f}")
        
        print(f"{'─'*80}\n")
        
        # Log epoch-level metrics to wandb
        if wandb.run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': avg_train['total_loss'],
                'train/reconstruction_loss': avg_train['reconstruction_loss'],
                'train/l1_loss': avg_train['l1_loss'],
                'train/l0_sparsity': avg_train['l0_sparsity'],
            }
            if val_metrics_list:
                log_dict.update({
                    'val/total_loss': avg_val['total_loss'],
                    'val/reconstruction_loss': avg_val['reconstruction_loss'],
                    'val/l1_loss': avg_val['l1_loss'],
                    'val/l0_sparsity': avg_val['l0_sparsity'],
                    'val/r2_score': avg_val['r2_score'],
                })
            wandb.log(log_dict, commit=True)
