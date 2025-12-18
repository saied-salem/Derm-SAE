import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse

def print_statistics(features):
    """Print comprehensive statistics of features."""
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(f"Shape: {features.shape}")
    print(f"  Images: {features.shape[0]}")
    print(f"  Patches per image: {features.shape[1]}")
    print(f"  Feature dimension: {features.shape[2]}")
    
    flat = features.flatten()
    print(f"\nValue Statistics:")
    print(f"  Mean: {flat.mean().item():.6f}")
    print(f"  Std: {flat.std().item():.6f}")
    print(f"  Min: {flat.min().item():.6f}")
    print(f"  Max: {flat.max().item():.6f}")
    print(f"  Median: {flat.median().item():.6f}")
    
    dim_means = features.mean(dim=(0, 1))
    dim_stds = features.std(dim=(0, 1))
    print(f"\nPer-dimension statistics:")
    print(f"  Mean range: [{dim_means.min().item():.6f}, {dim_means.max().item():.6f}]")
    print(f"  Std range: [{dim_stds.min().item():.6f}, {dim_stds.max().item():.6f}]")
    print("="*60 + "\n")

def resize_features_batch(features, target_size=224, batch_size=50):
    """Resize feature maps using bilinear interpolation in batches."""
    N, num_patches, D = features.shape
    H = W = int(num_patches ** 0.5)
    
    print(f"Resizing from {H}x{W} to {target_size}x{target_size} in batches of {batch_size}...")
    
    all_resized = []
    for i in tqdm(range(0, N, batch_size), desc="Resizing batches"):
        batch = features[i:i+batch_size]
        
        # Reshape to [B, D, H, W]
        batch_spatial = batch.permute(0, 2, 1).reshape(-1, D, H, W)
        
        # Interpolate [B, D, target_size, target_size]
        resized = F.interpolate(batch_spatial, size=(target_size, target_size), 
                               mode='bilinear', align_corners=False)
        
        # Back to [B, target_size*target_size, D]
        resized_patches = resized.reshape(-1, D, target_size*target_size).permute(0, 2, 1)
        all_resized.append(resized_patches.cpu())
        
        # Clear cache
        del batch, batch_spatial, resized, resized_patches
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    resized_features = torch.cat(all_resized, dim=0)
    print(f"Resized shape: {resized_features.shape}\n")
    return resized_features, (target_size, target_size)

def save_random_feature_maps(features, spatial_size, num_samples, num_channels, output_dir):
    """Save random feature map channels for each sample (memory efficient)."""
    os.makedirs(output_dir, exist_ok=True)
    
    N, num_patches, D = features.shape
    H, W = spatial_size
    
    print(f"Saving {num_channels} random feature maps for {num_samples} samples...")
    
    for sample_idx in tqdm(range(min(num_samples, N)), desc="Saving samples"):
        # Process one sample at a time
        sample = features[sample_idx].reshape(H, W, D).numpy()  # [H, W, D]
        
        # Randomly select channels
        random_channels = np.random.choice(D, size=min(num_channels, D), replace=False)
        
        # Create grid: 4 columns
        ncols = 4
        nrows = (len(random_channels) + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
        axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
        
        fig.suptitle(f'Sample {sample_idx} - Random Feature Maps', fontsize=16, y=0.995)
        
        for i, ch_idx in enumerate(random_channels):
            feature_map = sample[:, :, ch_idx]
            
            im = axes[i].imshow(feature_map, cmap='viridis', aspect='auto')
            axes[i].set_title(f'Channel {ch_idx}', fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(len(random_channels), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'sample_{sample_idx:03d}_feature_maps.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        del sample, fig, axes
    
    print(f"✓ Saved to {output_dir}/\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze and save feature maps")
    parser.add_argument('--input', type=str, default='features/dermlip_patch_features_val.pt',
                       help='Input features file')
    parser.add_argument('--output_features', type=str, default='features/dermlip_patch_features_val_224x224.pt',
                       help='Output file for resized features')
    parser.add_argument('--output_dir', type=str, default='feature_visualizations',
                       help='Directory to save feature map images')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target size for resizing (default: 224x224)')
    parser.add_argument('--max_images', type=int, default=100,
                       help='Maximum number of random images to process')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--num_channels', type=int, default=20,
                       help='Number of random channels to save per sample')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for resizing (lower = less memory)')
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features from {args.input}...")
    data = torch.load(args.input)
    all_features = data['patch_features'].cpu()
    all_labels = data['labels']
    del data
    
    N = all_features.shape[0]
    print(f"Total images in dataset: {N}")
    
    # Sample random subset
    num_to_sample = min(args.max_images, N)
    print(f"Randomly sampling {num_to_sample} images...")
    random_indices = torch.randperm(N)[:num_to_sample]
    features = all_features[random_indices]
    labels = all_labels[random_indices]
    del all_features, all_labels  # Free memory immediately
    
    print(f"Processing {features.shape[0]} images\n")
    
    # Show statistics
    print_statistics(features)
    
    # Resize features in batches (memory efficient)
    resized_features, spatial_size = resize_features_batch(features, args.target_size, args.batch_size)
    del features  # Free original features memory
    
    # Save resized features
    # print(f"Saving resized features to {args.output_features}...")
    # torch.save({
    #     'patch_features': resized_features,
    #     'labels': labels,
    #     'spatial_size': spatial_size
    # }, args.output_features)
    # print(f"✓ Saved\n")
    
    # Save random feature maps for samples
    save_random_feature_maps(resized_features, spatial_size, 
                            args.num_samples, args.num_channels, args.output_dir)
    
    print("="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

