"""
Step 3: Train Sparse Autoencoder on HAM10000 Patch Features
Following Mammo-SAE methodology with your existing sparse_autoencoder library
"""

import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("sparse_autoencoder/")
sys.path.append("dncbm/")

import os
from pathlib import Path
import torch
import numpy as np
import math
import datetime
import argparse
from time import time
from tqdm import tqdm

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
import wandb

try:
    from dncbm.custom_pipeline import Pipeline
except ImportError:
    print("Warning: dncbm.custom_pipeline not found. Using custom simplified pipeline.")
    Pipeline = None


CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma',
    'Benign keratosis-like lesions', 'Dermatofibroma',
    'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
]


def get_args():
    parser = argparse.ArgumentParser(description="Step 3: Train Sparse Autoencoder")
    
    # --- Input ---
    parser.add_argument('--train_feature_file', type=str, 
                       default='./dermlip_patch_features_train.pt',
                       help='Path to train patch features from Step 2')
    parser.add_argument('--val_feature_file', type=str,
                       default='./dermlip_patch_features_val.pt',
                       help='Path to val patch features from Step 2')
    
    # --- Output directories ---
    parser.add_argument('--output_dir', type=str, 
                       default='./sae_training_output',
                       help='Base output directory')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./sae_checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save_suffix', type=str, 
                       default='',
                       help='Suffix for save directories')
    
    # --- SAE Architecture ---
    parser.add_argument('--input_dim', type=int, 
                       default=768,
                       help='Input feature dimension (PanDerm: 768)')
    parser.add_argument('--expansion_factor', type=int, 
                       default=8,
                       help='Latent dimension = input_dim * expansion_factor')
    parser.add_argument('--n_components', type=int, 
                       default=1,
                       help='Number of SAE components (usually 1)')
    
    # --- Training hyperparameters ---
    parser.add_argument('--num_epochs', type=int, 
                       default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, 
                       default=4096,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, 
                       default=3e-4,
                       help='Learning rate')
    parser.add_argument('--l1_coeff', type=float, 
                       default=3e-5,
                       help='L1 sparsity penalty coefficient (lambda)')
    
    # --- Adam optimizer ---
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--adam_weight_decay', type=float, default=0.0)
    
    # --- Resampling (dead neuron handling) ---
    parser.add_argument('--resample_freq', type=int, 
                       default=10,
                       help='Resample dead neurons every N epochs')
    parser.add_argument('--resample_dataset_size', type=int, 
                       default=100000,
                       help='Number of samples for resampling (must be <= patches_per_file)')
    parser.add_argument('--patches_per_file', type=int,
                       default=100000,
                       help='Number of patches per activation file (default: 100000)')
    
    # --- Checkpointing & Validation ---
    parser.add_argument('--checkpoint_freq', type=int, 
                       default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', type=int, 
                       default=1,
                       help='Validate every N epochs (0 to disable)')
    
    # --- Logging ---
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, 
                       default='HAM10000-SAE',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, 
                       default=None,
                       help='W&B entity name')
    parser.add_argument('--experiment_name', type=str, 
                       default='panderm_sae',
                       help='Experiment name')
    
    # --- Device ---
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def prepare_activation_files(train_feature_file, val_feature_file, output_dir, 
                             patches_per_file=100000):
    """
    Convert patch features to activation files expected by the pipeline
    
    Loads train and val features separately from their respective files.
    
    SIMPLIFIED: Only save activations tensor, no metadata
    """
    
    print("="*80)
    print("Preparing activation files for SAE training...")
    print("="*80)
    
    # Load train patch features
    print(f"\nLoading train patch features from: {train_feature_file}")
    train_data = torch.load(train_feature_file)
    train_patch_features = train_data['patch_features']  # [N_train_images, 196, 768]
    train_labels = train_data['labels']  # [N_train_images]
    
    N_train_images, N_patches, D = train_patch_features.shape
    print(f"Loaded train: {N_train_images} images, {N_patches} patches/image, {D} features")
    
    # Load val patch features (if available)
    if val_feature_file and os.path.exists(val_feature_file):
        print(f"\nLoading val patch features from: {val_feature_file}")
        val_data = torch.load(val_feature_file)
        val_patch_features = val_data['patch_features']  # [N_val_images, 196, 768]
        val_labels = val_data['labels']  # [N_val_images]
        
        N_val_images = val_patch_features.shape[0]
        assert val_patch_features.shape[1] == N_patches, "Patch count mismatch between train and val!"
        assert val_patch_features.shape[2] == D, "Feature dimension mismatch between train and val!"
        print(f"Loaded val: {N_val_images} images, {N_patches} patches/image, {D} features")
    else:
        print(f"\nNo val features found at {val_feature_file}. Using train only.")
        val_patch_features = None
        val_labels = None
        N_val_images = 0
    
    # Flatten train features: [N_train_images, N_patches, D] -> [N_train_images * N_patches, D]
    train_patch_features_flat = train_patch_features.reshape(-1, D)
    
    # Normalize features using train set statistics (to avoid data leakage)
    print("\nNormalizing features (using train set statistics)...")
    mean = train_patch_features_flat.mean(dim=0, keepdim=True)
    std = train_patch_features_flat.std(dim=0, keepdim=True)
    
    train_patch_features_normalized = (train_patch_features_flat - mean) / (std + 1e-8)
    
    print(f"  Train - Original Mean: {train_patch_features_flat.mean():.4f}, Std: {train_patch_features_flat.std():.4f}")
    print(f"  Train - Normalized Mean: {train_patch_features_normalized.mean():.4f}, Std: {train_patch_features_normalized.std():.4f}")
    
    if val_patch_features is not None:
        val_patch_features_flat = val_patch_features.reshape(-1, D)
        val_patch_features_normalized = (val_patch_features_flat - mean) / (std + 1e-8)
        print(f"  Val - Normalized Mean: {val_patch_features_normalized.mean():.4f}, Std: {val_patch_features_normalized.std():.4f}")
    
    print(f"\n  Train patches: {len(train_patch_features_flat):,} ({N_train_images} images)")
    if val_patch_features is not None:
        print(f"  Val patches: {len(val_patch_features_flat):,} ({N_val_images} images)")
    else:
        print(f"  Val patches: 0 (no validation set)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verify write permissions
    try:
        test_file = output_path / '.write_test'
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        print(f"\n  ERROR: Cannot write to output directory: {output_path}")
        print(f"  Error: {e}")
        raise
    
    # Clean up old activation files and temp files
    print("\n  Cleaning up old activation files...")
    old_train_files = list(output_path.glob('train_*.pt'))
    old_val_files = list(output_path.glob('train_val_*.pt'))
    temp_files = list(output_path.glob('*.tmp'))  # Clean up any leftover temp files
    
    if old_train_files or old_val_files or temp_files:
        print(f"    Found {len(old_train_files)} old train files, {len(old_val_files)} old val files, {len(temp_files)} temp files")
        print(f"    Removing old files...")
        removed_count = 0
        for f in old_train_files + old_val_files + temp_files:
            try:
                if f.exists():  # Check if file still exists
                    f.unlink()
                    removed_count += 1
            except (FileNotFoundError, OSError) as e:
                # File might have been deleted already or other error
                continue
        print(f"    ✓ Cleaned up {removed_count} files")
    else:
        print(f"    No old files to clean up")
    
    # Save normalization stats (for later analysis)
    torch.save({
        'mean': mean,
        'std': std,
        'n_train_images': N_train_images,
        'n_val_images': N_val_images,
        'n_patches_per_image': N_patches,
        'feature_dim': D,
        'spatial_grid': (14, 14)
    }, output_path / 'normalization_stats.pt')
    
    # Helper to save chunks - ONLY ACTIVATIONS, NO METADATA
    def save_chunks(features_normalized, prefix):
        total_patches = len(features_normalized)
        n_files = (total_patches + patches_per_file - 1) // patches_per_file
        print(f"\n  Saving {prefix} data in {n_files} files...")
        
        saved_files = []
        for file_idx in tqdm(range(n_files), desc=f"  {prefix}"):
            start_idx = file_idx * patches_per_file
            end_idx = min(start_idx + patches_per_file, total_patches)
            
            # Extract chunk - just the activations
            chunk_features = features_normalized[start_idx:end_idx]
            
            # Save ONLY activations tensor - shape [B, 768]
            filename = output_path / f"{prefix}_{file_idx:03d}.pt"
            
            try:
                # Ensure directory exists
                filename.parent.mkdir(parents=True, exist_ok=True)
                
                # Check available disk space
                import shutil
                total, used, free = shutil.disk_usage(filename.parent)
                chunk_size_bytes = chunk_features.element_size() * chunk_features.numel()
                chunk_size_gb = chunk_size_bytes / (1024**3)
                
                # Warn if low on space (need at least 3x chunk size for safety)
                if free < chunk_size_bytes * 3:
                    print(f"\n  ⚠️  WARNING: Low disk space!")
                    print(f"     Free: {free / (1024**3):.2f} GB")
                    print(f"     Need: ~{chunk_size_gb * 3:.2f} GB for safe write")
                
                # Save with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Move tensor to CPU if needed and ensure contiguous
                        if chunk_features.is_cuda:
                            chunk_features = chunk_features.cpu()
                        if not chunk_features.is_contiguous():
                            chunk_features = chunk_features.contiguous()
                        
                        # Save to temporary file first, then rename (atomic write)
                        temp_filename = filename.with_suffix('.tmp')
                        
                        # Clean up any existing temp file
                        if temp_filename.exists():
                            temp_filename.unlink()
                        
                        torch.save(chunk_features, temp_filename)
                        
                        # Atomic rename
                        import os
                        os.replace(str(temp_filename), str(filename))
                        
                        saved_files.append(filename)
                        break  # Success, exit retry loop
                        
                    except (RuntimeError, OSError, IOError) as e:
                        if attempt < max_retries - 1:
                            import time
                            wait_time = (attempt + 1) * 2
                            print(f"\n  ⚠️  Write failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            # Clear cache if possible
                            import gc
                            gc.collect()
                        else:
                            # Final attempt failed
                            print(f"\n  ❌ ERROR: Failed to save {filename} after {max_retries} attempts")
                            print(f"     Error: {e}")
                            print(f"     Free disk space: {free / (1024**3):.2f} GB")
                            print(f"     Chunk size: {chunk_size_gb:.2f} GB")
                            raise RuntimeError(f"Failed to save activation file after {max_retries} attempts: {e}")
                
            except Exception as e:
                print(f"\n  ❌ FATAL ERROR saving {filename}: {e}")
                raise
        
        return saved_files
    
    # Save files separately
    saved_train_files = save_chunks(train_patch_features_normalized, 'train')
    if val_patch_features is not None:
        saved_val_files = save_chunks(val_patch_features_normalized, 'train_val')
    else:
        saved_val_files = []
    
    print("\n" + "="*80)
    print("✓ Activation files prepared successfully!")
    print(f"✓ Output directory: {output_path}")
    print(f"✓ Train files: {len(saved_train_files)}")
    if saved_val_files:
        print(f"✓ Val files: {len(saved_val_files)}")
    else:
        print(f"✓ Val files: 0 (no validation set)")
    print(f"✓ Total patches per file: {patches_per_file:,}")
    print("="*80 + "\n")
    
    return str(output_path)

def setup_wandb(args, start_time):
    """Initialize Weights & Biases logging"""
    if not args.use_wandb:
        return
    
    run_name = f"{args.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb_dir = os.path.join(args.output_dir, ".wandb_cache")
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        dir=wandb_dir,
        config=vars(args)
    )
    
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    
    print(f"✓ W&B initialized at {time() - start_time:.2f}s")


class SimplifiedPipeline:
    """
    Simplified training pipeline if dncbm.Pipeline is not available
    """
    
    def __init__(self, autoencoder, loss, optimizer, activation_resampler, 
                 checkpoint_directory, device, args):
        self.autoencoder = autoencoder
        self.loss = loss
        self.optimizer = optimizer
        self.activation_resampler = activation_resampler
        self.checkpoint_directory = Path(checkpoint_directory)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.args = args
        
    def load_activation_batch(self, file_paths, batch_size):
        """Load and batch activations from files"""
        all_activations = []
        all_metadata = []
        
        for file_path in file_paths:
            data = torch.load(file_path)
            all_activations.append(data['activations'])
            all_metadata.append(data['metadata'])
        
        # Concatenate all
        activations = torch.cat(all_activations, dim=0)
        
        # Create batches
        n_samples = len(activations)
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield activations[batch_indices]
    
    def train_epoch(self, train_files, batch_size, epoch):
        """Train for one epoch"""
        self.autoencoder.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        l0_activations = []
        n_batches = 0
        
        pbar = tqdm(self.load_activation_batch(train_files, batch_size), 
                   desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Add component dimension if needed: [B, D] -> [B, 1, D]
            if batch.dim() == 2 and self.autoencoder.n_components > 0:
                batch = batch.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            sae_out = self.autoencoder(batch)
            learned_activations = sae_out.learned_activations
            reconstructed = sae_out.decoded_activations
            
            # Compute loss
            loss_dict = self.loss.forward(
                learned_activations=learned_activations,
                reconstructed_activations=reconstructed,
                source_activations=batch
            )
            
            total_batch_loss = loss_dict['loss']
            
            # Backward
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += total_batch_loss.item()
            total_recon_loss += loss_dict.get('reconstruction_loss', 0)
            total_sparsity_loss += loss_dict.get('l1_loss', 0)
            
            # L0 sparsity
            l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
            l0_activations.append(l0)
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'L0': f'{l0:.1f}'
            })
        
        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_sparsity_loss': total_sparsity_loss / n_batches,
            'train_l0': np.mean(l0_activations)
        }
    
    def validate(self, val_files, batch_size):
        """Validate the model"""
        self.autoencoder.eval()
        
        total_loss = 0
        total_recon_loss = 0
        l0_activations = []
        r2_scores = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.load_activation_batch(val_files, batch_size):
                batch = batch.to(self.device)
                
                if batch.dim() == 2 and self.autoencoder.n_components > 0:
                    batch = batch.unsqueeze(1)
                
                sae_out = self.autoencoder(batch)
                learned_activations = sae_out.learned_activations
                reconstructed = sae_out.decoded_activations
                
                loss_dict = self.loss.forward(
                    learned_activations=learned_activations,
                    reconstructed_activations=reconstructed,
                    source_activations=batch
                )
                
                total_loss += loss_dict['loss'].item()
                total_recon_loss += loss_dict.get('reconstruction_loss', 0)
                
                l0 = (learned_activations > 0).float().sum(dim=-1).mean().item()
                l0_activations.append(l0)
                
                # R² score
                if reconstructed.dim() == 3:
                    reconstructed = reconstructed.squeeze(1)
                    batch = batch.squeeze(1)
                
                ss_res = ((batch - reconstructed) ** 2).sum()
                ss_tot = ((batch - batch.mean(dim=0)) ** 2).sum()
                r2 = 1 - ss_res / ss_tot
                r2_scores.append(r2.item())
                
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
            'val_l0': np.mean(l0_activations),
            'val_r2': np.mean(r2_scores)
        }
    
    def save_checkpoint(self, epoch, metrics, filename='checkpoint.pt'):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_directory / filename
        
        torch.save({
            'epoch': epoch,
            'state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.args)
        }, checkpoint_path)
        
        return checkpoint_path
    
    def run_pipeline(self, train_batch_size, checkpoint_frequency, val_frequency,
                    num_epochs, train_fnames, train_val_fnames, start_time, 
                    resample_epoch_freq):
        """Run the full training pipeline"""
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(train_fnames, train_batch_size, epoch)
            
            # Validate
            val_metrics = {}
            if val_frequency > 0 and (epoch + 1) % val_frequency == 0 and train_val_fnames:
                val_metrics = self.validate(train_val_fnames, train_batch_size)
                print(f"\nValidation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"R²: {val_metrics['val_r2']:.4f}, L0: {val_metrics['val_l0']:.1f}")
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            
            print(f"\nTrain - Loss: {train_metrics['train_loss']:.4f}, "
                  f"L0: {train_metrics['train_l0']:.1f}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log(metrics)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                ckpt_path = self.save_checkpoint(
                    epoch, metrics, f'checkpoint_epoch_{epoch+1}.pt'
                )
                print(f"✓ Saved checkpoint: {ckpt_path}")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                ckpt_path = self.save_checkpoint(epoch, metrics, 'best_checkpoint.pt')
                print(f"✓ Saved best model: {ckpt_path}")
        
        # Save final checkpoint
        final_path = self.save_checkpoint(num_epochs-1, metrics, 'final_checkpoint.pt')
        print(f"\n✓ Training complete! Final checkpoint: {final_path}")


def main():
    args = get_args()
    start_time = time()
    
    print("="*80)
    print("Step 3: Train Sparse Autoencoder")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Expansion factor: {args.expansion_factor}")
    print(f"Latent dimension: {args.input_dim * args.expansion_factor}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"L1 coefficient: {args.l1_coeff}")
    print("="*80 + "\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare activation files
    activation_dir = prepare_activation_files(
        train_feature_file=args.train_feature_file,
        val_feature_file=args.val_feature_file,
        output_dir=os.path.join(args.output_dir, 'activations'),
        patches_per_file=args.patches_per_file
    )
    
    # Get file lists
    train_fnames = sorted([str(p) for p in Path(activation_dir).glob('train_*.pt') 
                          if not p.name.startswith('train_val')])
    train_val_fnames = sorted([str(p) for p in Path(activation_dir).glob('train_val_*.pt')])
    
    if args.val_freq == 0:
        train_fnames = train_fnames + train_val_fnames
        train_val_fnames = None
    
    print(f"Train files: {len(train_fnames)}")
    if train_val_fnames:
        print(f"Val files: {len(train_val_fnames)}\n")
    
    # Create SAE
    n_learned_features = args.input_dim * args.expansion_factor
    
    autoencoder = SparseAutoencoder(
        n_input_features=args.input_dim,
        n_learned_features=n_learned_features,
        n_components=args.n_components,
    ).to(args.device)
    
    print(f"✓ SAE created at {time() - start_time:.2f}s")
    print(f"  Parameters: {sum(p.numel() for p in autoencoder.parameters()):,}\n")
    
    # Create loss
    loss = LossReducer(
        LearnedActivationsL1Loss(l1_coefficient=args.l1_coeff),
        L2ReconstructionLoss(),
    )
    print(f"✓ Loss created at {time() - start_time:.2f}s")
    
    # Create optimizer
    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=args.lr,
        betas=(args.adam_beta_1, args.adam_beta_2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
        has_components_dim=(args.n_components > 0),
    )
    print(f"✓ Optimizer created at {time() - start_time:.2f}s")
    
    # Create activation resampler
    activation_resampler = ActivationResampler(
        resample_interval=1,
        n_activations_activity_collate=1,
        max_n_resamples=math.inf,
        n_learned_features=n_learned_features,
        resample_epoch_freq=args.resample_freq,
        resample_dataset_size=args.resample_dataset_size,
    )
    print(f"✓ Activation resampler created at {time() - start_time:.2f}s")
    
    # Setup wandb
    setup_wandb(args, start_time)
    
    # Create pipeline
    checkpoint_dir = Path(args.checkpoint_dir + args.save_suffix)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if Pipeline is not None:
        pipeline = Pipeline(
            activation_resampler=activation_resampler,
            autoencoder=autoencoder,
            checkpoint_directory=checkpoint_dir,
            loss=loss,
            optimizer=optimizer,
            device=args.device,
            args=args,
        )
    else:
        pipeline = SimplifiedPipeline(
            autoencoder=autoencoder,
            loss=loss,
            optimizer=optimizer,
            activation_resampler=activation_resampler,
            checkpoint_directory=checkpoint_dir,
            device=args.device,
            args=args,
        )
    
    print(f"✓ Pipeline created at {time() - start_time:.2f}s\n")
    
    # Run training
    print("="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    pipeline.run_pipeline(
        train_batch_size=args.batch_size,
        checkpoint_frequency=args.checkpoint_freq,
        val_frequency=args.val_freq,
        num_epochs=args.num_epochs,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=args.resample_freq,
    )
    
    total_time = time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ Total training time: {total_time/60:.2f} minutes")
    print(f"✓ Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*80}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()