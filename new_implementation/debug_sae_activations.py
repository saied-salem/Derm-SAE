"""
Quick diagnostic script to check if SAE produces non-zero activations
"""
import sys
sys.path.append("sparse_autoencoder/")

import torch
from sparse_autoencoder import SparseAutoencoder
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_checkpoint', type=str, required=True)
    parser.add_argument('--normalization_stats', type=str, required=True)
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("="*80)
    print("SAE Activation Diagnostic")
    print("="*80)
    
    # Load normalization stats
    print("\n1. Loading normalization stats...")
    norm_stats = torch.load(args.normalization_stats, map_location='cpu')
    feature_mean = norm_stats['mean']
    feature_std = norm_stats['std']
    print(f"   Mean shape: {feature_mean.shape}")
    print(f"   Mean range: [{feature_mean.min().item():.6f}, {feature_mean.max().item():.6f}]")
    print(f"   Std range: [{feature_std.min().item():.6f}, {feature_std.max().item():.6f}]")
    
    # Load SAE
    print("\n2. Loading SAE...")
    n_learned_features = args.input_dim * args.expansion_factor
    sae = SparseAutoencoder(
        n_input_features=args.input_dim,
        n_learned_features=n_learned_features,
        n_components=args.n_components,
    ).to(args.device)
    
    checkpoint = torch.load(args.sae_checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
        print(f"   Loaded from 'state_dict' key")
        if 'epoch' in checkpoint:
            print(f"   Checkpoint epoch: {checkpoint['epoch']}")
    else:
        sae.load_state_dict(checkpoint)
        print(f"   Loaded directly")
    sae.eval()
    print(f"   SAE has {sum(p.numel() for p in sae.parameters()):,} parameters")
    
    # Test 1: Random normalized input (should work)
    print("\n3. Test 1: Random normalized input (mean=0, std=1)")
    random_input = torch.randn(196, args.input_dim).to(args.device)
    if args.n_components > 0:
        random_input = random_input.unsqueeze(1)
    
    with torch.no_grad():
        sae_out = sae(random_input)
        activations = sae_out.learned_activations
        if activations.dim() == 3:
            activations = activations.squeeze(1)
        
        print(f"   Output shape: {activations.shape}")
        print(f"   Mean: {activations.mean().item():.6f}, Std: {activations.std().item():.6f}")
        print(f"   Min: {activations.min().item():.6f}, Max: {activations.max().item():.6f}")
        print(f"   Non-zero: {(activations > 0).sum().item()} / {activations.numel()}")
        print(f"   Sparsity (L0): {(activations > 0).float().sum(dim=-1).mean().item():.1f}")
    
    # Test 2: Simulated "real" features with normalization
    print("\n4. Test 2: Simulated features (uniform [0, 1]) + normalization")
    simulated_features = torch.rand(196, args.input_dim).to(args.device)
    normalized_features = (simulated_features - feature_mean.to(args.device)) / (feature_std.to(args.device) + 1e-8)
    
    print(f"   Before normalization - Mean: {simulated_features.mean().item():.6f}, Std: {simulated_features.std().item():.6f}")
    print(f"   After normalization - Mean: {normalized_features.mean().item():.6f}, Std: {normalized_features.std().item():.6f}")
    
    if args.n_components > 0:
        normalized_features = normalized_features.unsqueeze(1)
    
    with torch.no_grad():
        sae_out = sae(normalized_features)
        activations = sae_out.learned_activations
        if activations.dim() == 3:
            activations = activations.squeeze(1)
        print(" sae_out:", sae_out.shape)
        print(f"   Output shape: {activations.shape}")
        print(f"   Mean: {activations.mean().item():.6f}, Std: {activations.std().item():.6f}")
        print(f"   Min: {activations.min().item():.6f}, Max: {activations.max().item():.6f}")
        print(f"   Non-zero: {(activations > 0).sum().item()} / {activations.numel()}")
        print(f"   Sparsity (L0): {(activations > 0).float().sum(dim=-1).mean().item():.1f}")
    
    # Test 3: Check encoder weights
    print("\n5. Checking encoder weights...")
    encoder_weight = sae._encoder.weight  # Should be [n_learned_features, n_input_features]
    print(f"   Encoder weight shape: {encoder_weight.shape}")
    print(f"   Mean: {encoder_weight.mean().item():.6f}, Std: {encoder_weight.std().item():.6f}")
    print(f"   Min: {encoder_weight.min().item():.6f}, Max: {encoder_weight.max().item():.6f}")
    
    if hasattr(sae._encoder, 'bias') and sae._encoder.bias is not None:
        encoder_bias = sae._encoder.bias
        print(f"   Encoder bias shape: {encoder_bias.shape}")
        print(f"   Mean: {encoder_bias.mean().item():.6f}, Std: {encoder_bias.std().item():.6f}")
        print(f"   Min: {encoder_bias.min().item():.6f}, Max: {encoder_bias.max().item():.6f}")
    else:
        print(f"   No encoder bias")
    
    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80)

if __name__ == "__main__":
    main()

