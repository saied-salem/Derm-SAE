"""
Step 4A: Find Top Concepts Per Class for HAM10000 (CORRECTED VERSION)

This script DIRECTLY uses the patch features from Step 2 (no re-extraction needed!)
Following Mammo-SAE methodology exactly:
  1. Load patch features from Step 2 files
  2. Pass through trained SAE to get concepts
  3. Spatially average concepts per image
  4. Rank concepts by class importance
"""

import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("sparse_autoencoder/")
# sys.path.append("dncbm/")


import os
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from sparse_autoencoder import SparseAutoencoder

CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma',
    'Benign keratosis-like lesions', 'Dermatofibroma',
    'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Step 4A: Find top concepts per class (uses existing patch features)"
    )
    
    # ============================================
    # INPUT: Patch features from Step 2
    # ============================================
    parser.add_argument('--patch_features_file', type=str, required=True,
                       help='Path to patch features file from Step 2 (e.g., dermlip_patch_features_val.pt)')
    
    # ============================================
    # INPUT: Trained SAE from Step 3
    # ============================================
    parser.add_argument('--sae_checkpoint', type=str, required=True,
                       help='Path to trained SAE checkpoint from Step 3')
    parser.add_argument('--normalization_stats', type=str, required=True,
                       help='Path to normalization_stats.pt from Step 3 training')
    
    # ============================================
    # SAE Architecture (must match Step 3 training)
    # ============================================
    parser.add_argument('--input_dim', type=int, default=768,
                       help='SAE input dimension (PanDerm: 768)')
    parser.add_argument('--expansion_factor', type=int, default=8,
                       help='SAE expansion factor (must match training)')
    parser.add_argument('--n_components', type=int, default=1,
                       help='Number of SAE components (must match training)')
    
    # ============================================
    # Processing settings
    # ============================================
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for processing images')
    parser.add_argument('--split_name', type=str, default='val',
                       help='Name of this split (e.g., val, train)')
    
    # ============================================
    # Output
    # ============================================
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Output directory for results')
    
    # ============================================
    # Device
    # ============================================
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()


def extract_concepts_from_patches(patch_features, sae, args, feature_mean, feature_std, batch_size=128):
    """
    Extract SAE concepts from existing patch features
    
    Args:
        patch_features: [N_images, 196, 768] tensor from Step 2
        sae: Trained SparseAutoencoder
        args: Arguments
        feature_mean: [1, 768] mean from training
        feature_std: [1, 768] std from training
        batch_size: Batch size for processing
    
    Returns:
        concepts_per_image: [N_images, n_learned_features] averaged concepts
    """
    print("\nExtracting SAE concepts from patch features...")
    
    N_images, N_patches, D = patch_features.shape
    print(f"  Input: {N_images} images, {N_patches} patches/image, {D} features")
    
    n_learned_features = args.input_dim * args.expansion_factor
    all_concepts = []
    
    sae.eval()
    with torch.no_grad():
        # Process images in batches
        for batch_start in tqdm(range(0, N_images, batch_size), desc="Processing images"):
            batch_end = min(batch_start + batch_size, N_images)
            batch_patches = patch_features[batch_start:batch_end]  # [B, 196, 768]
            
            B = batch_patches.shape[0]
            H_patches = W_patches = 14  # 196 = 14x14
            
            # Reshape to spatial: [B, 196, 768] -> [B, 14, 14, 768]
            batch_patches = batch_patches.reshape(B, H_patches, W_patches, D)
            
            # Flatten for SAE: [B, 14, 14, 768] -> [B*196, 768]
            batch_patches_flat = batch_patches.reshape(-1, D).to(args.device)
            
            # CRITICAL: Normalize features using training statistics
            batch_patches_normalized = (batch_patches_flat - feature_mean.to(args.device)) / (feature_std.to(args.device) + 1e-8)
            
            # Add component dimension if needed: [B*196, 768] -> [B*196, 1, 768]
            if args.n_components > 0:
                batch_patches_normalized = batch_patches_normalized.unsqueeze(1)
            
            # Pass through SAE (with normalized features!)
            sae_out = sae(batch_patches_normalized)
            
            concepts = sae_out.learned_activations  # [B*196, n_learned_features] or [B*196, 1, n_learned_features]
            
            if concepts.dim() == 3:
                concepts = concepts.squeeze(1)  # [B*196, n_learned_features]
            
            # Reshape back to spatial: [B*196, n_learned_features] -> [B, 14, 14, n_learned_features]
            concepts = concepts.reshape(B, H_patches, W_patches, -1)
            
            # Permute to [B, n_learned_features, 14, 14]
            concepts = concepts.permute(0, 3, 1, 2)
            
            # ⭐ CRITICAL: Spatial averaging (exactly like Mammo-SAE)
            # torch.mean(concepts, (2, 3)) -> [B, n_learned_features]
            concepts_avg = torch.mean(concepts, dim=(2, 3))
            
            all_concepts.append(concepts_avg.cpu())
    
    # Concatenate all batches: [N_images, n_learned_features]
    concepts_per_image = torch.cat(all_concepts, dim=0)
    
    print(f"✓ Extracted concepts: {concepts_per_image.shape}")
    return concepts_per_image


def save_global_concepts(patch_features_file, sae, args, feature_mean, feature_std):
    """
    STEP 1: Load patch features and extract global concept strengths
    Following Mammo-SAE: save_concept_strengths_global.py
    """
    print("\n" + "="*80)
    print("STEP 1: Extracting Global Concept Strengths")
    print("="*80)
    print(f"Loading patch features from: {patch_features_file}")
    
    # Load patch features from Step 2
    data = torch.load(patch_features_file)
    patch_features = data['patch_features']  # [N_images, 196, 768]
    labels = data['labels']  # [N_images]
    
    N_images = len(labels)
    print(f"✓ Loaded {N_images} images")
    
    # Extract concepts using trained SAE (with normalization!)
    concepts_per_image = extract_concepts_from_patches(
        patch_features, sae, args, feature_mean, feature_std, batch_size=args.batch_size
    )
    
    # Save global concepts
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / f"sae_concepts_ham10000_{args.split_name}_global.pth"
    
    # Note: We don't have image paths in the patch features file,
    # so we'll create placeholder paths or use image indices
    img_paths = [f"image_{i:06d}" for i in range(N_images)]
    
    torch.save({
        'sae_concepts': concepts_per_image,
        'labels': labels,
        'img_paths': img_paths,
    }, save_path)
    
    print(f"\n✓ Saved global concepts to: {save_path}")
    print(f"  Shape: {concepts_per_image.shape}")
    print(f"  Labels: {labels.shape}")
    
    return save_path


def find_top_concepts_by_class(data, class_idx):
    """
    Find top concepts for a specific class
    Following Mammo-SAE: analyse_concepts.py
    """
    # Filter by class
    indexes = data['labels'] == class_idx
    sae_concepts_class = data['sae_concepts'][indexes]
    
    # Average across all images in this class
    concept_importance = torch.mean(sae_concepts_class, dim=0).abs()
    
    # Rank all concepts
    top_k_importance, top_k_indices = torch.topk(
        concept_importance,
        k=sae_concepts_class.shape[1]
    )
    
    # Format results
    result = []
    for rank, (index, importance) in enumerate(zip(top_k_indices, top_k_importance)):
        result.append({
            'rank': rank,
            'index': index.item(),
            'importance': importance.item()
        })
    
    return result


def find_top_concepts_classagnostic(data):
    """
    Find top concepts across all classes
    Following Mammo-SAE: analyse_concepts.py
    """
    sae_concepts = data['sae_concepts']
    
    # Average across ALL images
    concept_importance = torch.mean(sae_concepts, dim=0).abs()
    
    # Rank all concepts
    top_k_importance, top_k_indices = torch.topk(
        concept_importance,
        k=sae_concepts.shape[1]
    )
    
    result = []
    for rank, (index, importance) in enumerate(zip(top_k_indices, top_k_importance)):
        result.append({
            'rank': rank,
            'index': index.item(),
            'importance': importance.item()
        })
    
    return result


def analyse_concepts(global_concepts_path, args):
    """
    STEP 2: Analyze concepts to find top per class
    Following Mammo-SAE: analyse_concepts.py
    """
    print("\n" + "="*80)
    print("STEP 2: Analyzing Concepts Per Class")
    print("="*80)
    
    # Load global concepts
    print(f"Loading from: {global_concepts_path}")
    data = torch.load(global_concepts_path)
    
    print(f"  Total images: {len(data['labels'])}")
    print(f"  Concept dimension: {data['sae_concepts'].shape[1]}")
    
    # Print class distribution
    print("\n  Class distribution:")
    for class_idx in range(len(CLASS_NAMES)):
        n_images = (data['labels'] == class_idx).sum().item()
        print(f"    Class {class_idx} ({CLASS_NAMES[class_idx]}): {n_images} images")
    
    # Create output directory
    output_dir = Path(args.save_dir) / 'class_wise_concepts' / args.split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each class
    print("\n" + "-"*80)
    for class_idx in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_idx]
        n_images = (data['labels'] == class_idx).sum().item()
        
        print(f"\n{class_name} (Class {class_idx}): {n_images} images")
        
        if n_images == 0:
            print(f"  ⚠️  No images found")
            continue
        
        # Find top concepts
        result = find_top_concepts_by_class(data, class_idx)
        
        # Save
        save_path = output_dir / f'top_k_concepts_class={class_idx}.json'
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print top 10
        print(f"  Top 10 concepts:")
        for item in result[:10]:
            print(f"    {item['rank']+1:2d}. Concept {item['index']:4d}: "
                  f"importance={item['importance']:.4f}")
    
    # Class-agnostic
    print(f"\n" + "-"*80)
    print("Analyzing class-agnostic concepts...")
    result_all = find_top_concepts_classagnostic(data)
    
    save_path_all = output_dir / 'top_k_concepts_class=all.json'
    with open(save_path_all, 'w') as f:
        json.dump(result_all, f, indent=2)
    
    print(f"  Top 10 class-agnostic:")
    for item in result_all[:10]:
        print(f"    {item['rank']+1:2d}. Concept {item['index']:4d}: "
              f"importance={item['importance']:.4f}")
    
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    args = get_args()
    
    print("="*80)
    print("Step 4A: Find Top Concepts (Using Existing Features)")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Patch features: {args.patch_features_file}")
    print(f"SAE checkpoint: {args.sae_checkpoint}")
    print(f"Normalization stats: {args.normalization_stats}")
    print(f"Split: {args.split_name}")
    print("="*80)
    
    # Verify input files exist
    if not os.path.exists(args.patch_features_file):
        print(f"\n❌ ERROR: Patch features file not found: {args.patch_features_file}")
        print("Make sure you've run Step 2 first!")
        return
    
    if not os.path.exists(args.sae_checkpoint):
        print(f"\n❌ ERROR: SAE checkpoint not found: {args.sae_checkpoint}")
        print("Make sure you've run Step 3 first!")
        return
    
    if not os.path.exists(args.normalization_stats):
        print(f"\n❌ ERROR: Normalization stats not found: {args.normalization_stats}")
        print("Make sure you've run Step 3 first!")
        return
    
    # Load normalization stats
    print("\nLoading normalization stats...")
    norm_stats = torch.load(args.normalization_stats, map_location='cpu')
    feature_mean = norm_stats['mean']  # [1, 768]
    feature_std = norm_stats['std']    # [1, 768]
    print(f"✓ Loaded normalization stats")
    print(f"  Mean shape: {feature_mean.shape}, Std shape: {feature_std.shape}")
    
    # Load SAE
    print("\nLoading trained SAE...")
    n_learned_features = args.input_dim * args.expansion_factor
    sae = SparseAutoencoder(
        n_input_features=args.input_dim,
        n_learned_features=n_learned_features,
        n_components=args.n_components,
    ).to(args.device)
    
    checkpoint = torch.load(args.sae_checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint:
            print(f"✓ SAE loaded from epoch {checkpoint['epoch']}")
    else:
        sae.load_state_dict(checkpoint)
    
    print(f"✓ SAE loaded: {n_learned_features} learned features")
    
    # STEP 1: Extract and save global concepts
    global_concepts_path = save_global_concepts(
        args.patch_features_file, sae, args, feature_mean, feature_std
    )
    
    # STEP 2: Analyze concepts per class
    analyse_concepts(global_concepts_path, args)
    
    print(f"\n{'='*80}")
    print("✓✓✓ Step 4A Complete! ✓✓✓")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  1. Global concepts: {args.save_dir}/sae_concepts_ham10000_{args.split_name}_global.pth")
    print(f"  2. Class rankings: {args.save_dir}/class_wise_concepts/{args.split_name}/")
    print(f"\nNext: Run Step 4B (visualization) using these concept rankings")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()