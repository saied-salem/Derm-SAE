"""
Find which neurons are actually active on validation images
to ensure you're visualizing concepts that appear in your val set
"""
import sys
sys.path.append("sparse_autoencoder/")
sys.path.append("Derm1M")
sys.path.append("Derm1M/src")

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from sparse_autoencoder import SparseAutoencoder
from Derm1M.src.open_clip import create_model_and_transforms
import torch.nn as nn
import json

class FeatureExtractor:
    def __init__(self, model):
        self.patch_features = None
        hook_target = model.visual.blocks[-1]
        self.hook = hook_target.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.patch_features = output
    
    def get_features(self):
        return self.patch_features
    
    def close(self):
        self.hook.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_features', type=str, required=True,
                       help='Path to val patch features .pt file')
    parser.add_argument('--sae_checkpoint', type=str, required=True)
    parser.add_argument('--normalization_stats', type=str, required=True)
    parser.add_argument('--output_json', type=str, default='val_active_neurons.json')
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=50,
                       help='Number of top neurons to report')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("Finding Active Neurons on Validation Set")
    print("="*80)
    
    # Load features
    print(f"\n1. Loading validation features from {args.val_features}...")
    data = torch.load(args.val_features)
    patch_features = data['patch_features']  # [N, 196, 768]
    labels = data['labels']
    print(f"   Loaded {patch_features.shape[0]} images")
    
    # Load normalization stats
    print(f"\n2. Loading normalization stats...")
    norm_stats = torch.load(args.normalization_stats, map_location='cpu')
    feature_mean = norm_stats['mean']
    feature_std = norm_stats['std']
    print(f"   Mean shape: {feature_mean.shape}, Std shape: {feature_std.shape}")
    
    # Load SAE
    print(f"\n3. Loading SAE...")
    n_learned_features = args.input_dim * args.expansion_factor
    sae = SparseAutoencoder(
        n_input_features=args.input_dim,
        n_learned_features=n_learned_features,
        n_components=args.n_components,
    ).to(args.device)
    
    checkpoint = torch.load(args.sae_checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
    else:
        sae.load_state_dict(checkpoint)
    sae.eval()
    print(f"   SAE loaded with {n_learned_features} learned features")
    
    # Process all validation images
    print(f"\n4. Computing activations for all validation images...")
    all_activations = []
    
    # Flatten patches: [N, 196, 768] -> [N*196, 768]
    N, n_patches, D = patch_features.shape
    patch_features_flat = patch_features.reshape(-1, D)
    
    # Normalize
    patch_features_normalized = (patch_features_flat - feature_mean) / (feature_std + 1e-8)
    
    # Process in batches to avoid OOM
    batch_size = 4096
    with torch.no_grad():
        for i in tqdm(range(0, len(patch_features_normalized), batch_size), desc="Processing"):
            batch = patch_features_normalized[i:i+batch_size].to(args.device)
            
            if args.n_components > 0:
                batch = batch.unsqueeze(1)
            
            sae_out = sae(batch)
            activations = sae_out.learned_activations
            
            if activations.dim() == 3:
                activations = activations.squeeze(1)
            
            all_activations.append(activations.cpu())
    
    all_activations = torch.cat(all_activations, dim=0)  # [N*196, n_learned_features]
    print(f"   Total activations shape: {all_activations.shape}")
    
    # Compute statistics per neuron
    print(f"\n5. Computing neuron statistics...")
    
    # Mean activation per neuron
    mean_activations = all_activations.mean(dim=0)  # [n_learned_features]
    
    # Max activation per neuron
    max_activations = all_activations.max(dim=0)[0]  # [n_learned_features]
    
    # Frequency of activation (how often > 0)
    activation_freq = (all_activations > 0).float().mean(dim=0)  # [n_learned_features]
    
    # Find top-k neurons by mean activation
    top_k_indices = torch.argsort(mean_activations, descending=True)[:args.top_k]
    
    results = []
    for rank, idx in enumerate(top_k_indices):
        idx = idx.item()
        results.append({
            'rank': rank,
            'index': idx,
            'mean_activation': mean_activations[idx].item(),
            'max_activation': max_activations[idx].item(),
            'activation_frequency': activation_freq[idx].item()
        })
    
    # Save results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved top {args.top_k} active neurons to {output_path}")
    print(f"\nTop 10 most active neurons on validation set:")
    for i in range(min(10, len(results))):
        r = results[i]
        print(f"  {i+1}. Neuron {r['index']}: mean={r['mean_activation']:.4f}, "
              f"max={r['max_activation']:.4f}, freq={r['activation_frequency']:.4f}")
    
    print("\n" + "="*80)
    print("Complete! Use this JSON for visualization:")
    print(f"  --top_concepts_json {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

