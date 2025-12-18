"""
Visualize SAE Concepts on HAM10000 Images
Following Mammo-SAE visualization methodology (visualize_latent_neuron.py)

IMPORTANT: Features MUST be normalized using the same statistics from training
before passing to SAE, otherwise the learned weights won't match the input distribution.
"""

import sys
sys.path.append("../")
sys.path.append("./")
sys.path.append("Derm1M")
sys.path.append("Derm1M/src")
sys.path.append("sparse_autoencoder/")


import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import torch.nn.functional as F
import pickle
from skimage import measure

from sparse_autoencoder import SparseAutoencoder
from Derm1M.src.open_clip import create_model_and_transforms
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma',
    'Benign keratosis-like lesions', 'Dermatofibroma',
    'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
]


def get_args():
    parser = argparse.ArgumentParser(description="Visualize SAE concepts on HAM10000 (Mammo-SAE style)")
    
    # Model paths
    parser.add_argument('--sae_checkpoint', type=str, required=True,
                       help='Path to trained SAE checkpoint')
    parser.add_argument('--normalization_stats', type=str, required=True,
                       help='Path to normalization_stats.pt from training')
    parser.add_argument('--panderm_model', type=str, 
                       default='hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256')
    
    # Data
    parser.add_argument('--data_split_file', type=str, required=True,
                       help='Path to pickle file (class2images_val.p)')
    parser.add_argument('--image_root_dir', type=str, required=True,
                       help='Root directory of HAM10000 images')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    
    # SAE architecture
    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--expansion_factor', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=1)
    
    # Visualization settings
    parser.add_argument('--top_concepts_json', type=str, required=True,
                       help='Path to top_k_concepts JSON file from analysis')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Visualize top-k concepts from JSON')
    parser.add_argument('--num_images', type=int, default=20,
                       help='Number of images to visualize')
    parser.add_argument('--class_filter', type=int, default=None,
                       help='Only visualize images from this class (0-6)')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                       help='Relative threshold for binary mask (0-1)')
    parser.add_argument('--split_name', type=str, default='val')
    parser.add_argument('--save_dir', type=str, default='./results/visualization')
    
    # Device
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (keep at 1 for visualization)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    return parser.parse_args()


class HAM10000_Dataset(Dataset):
    """Dataset for visualization"""
    def __init__(self, pickle_split_file, image_root_dir, image_ext, processor, class_filter=None):
        with open(pickle_split_file, 'rb') as f:
            class2images = pickle.load(f)
        
        self.image_root_dir = Path(image_root_dir)
        self.processor = processor
        self.image_list = []
        
        lower_class_names = [name.lower() for name in CLASS_NAMES]
        for class_name, img_filenames in class2images.items():
            try:
                class_idx = lower_class_names.index(class_name.lower())
            except ValueError:
                continue
            
            # Filter by class if specified
            if class_filter is not None and class_idx != class_filter:
                continue
            
            for img_name in img_filenames:
                img_name_stem = img_name.split(".")[0]
                img_path = self.image_root_dir / f"{img_name_stem}{image_ext}"
                if img_path.exists():
                    self.image_list.append({
                        'path': str(img_path),
                        'label_idx': class_idx,
                        'image_id': img_name_stem
                    })
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        item = self.image_list[idx]
        img_path = item['path']
        label = item['label_idx']
        image_id = item['image_id']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))
        
        pixel_values = self.processor(image)
        return pixel_values, torch.tensor(label, dtype=torch.long), img_path, image_id


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


def visualize_topk_concepts(
    image_tensor, 
    concepts_spatial,
    label,
    img_path,
    image_id,
    latent_neurons,
    args
):
    """
    Visualize top-k concepts for an image in a grid layout
    Row 1: Original image (top left) + individual heatmaps
    Row 2: Heatmaps overlaid on images
    Row 3: Contours only
    
    Args:
        image_tensor: [1, 3, 224, 224] preprocessed image
        concepts_spatial: [14, 14, n_learned_features] spatial concept activations
        label: Ground truth label
        img_path: Path to original image
        image_id: Image ID
        latent_neurons: List of concept indices to visualize
        args: Arguments
    """
    # Load original image for better visualization
    original_image = Image.open(img_path).convert('RGB')
    img_np = np.array(original_image)
    
    H_patches, W_patches, _ = concepts_spatial.shape
    H_img, W_img = img_np.shape[:2]
    
    # Create output directory: visualization_{class_name}
    class_name = CLASS_NAMES[label]
    class_name_clean = class_name.replace(" ", "_").replace("-", "_")
    save_dir = Path(args.save_dir).parent / f"visualization_{class_name_clean}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    n_concepts = len(latent_neurons)
    
    # Create figure: 3 rows x (n_concepts + 1) columns
    fig, axes = plt.subplots(3, n_concepts + 1, figsize=(4 * (n_concepts + 1), 12))
    
    # Ensure axes is 2D
    if n_concepts == 0:
        return results
    
    # Row 1, Col 1: Original image
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f"Original Image\n{image_id}\nClass: {class_name}", fontsize=10)
    axes[0, 0].axis('off')
    
    # Hide other cells in first column
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    # Process each concept
    for col_idx, neuron_idx in enumerate(latent_neurons):
        # Get concept heatmap [14, 14]
        heatmap = concepts_spatial[:, :, neuron_idx].cpu().numpy()
        
        # Upsample to image resolution
        heatmap_upsampled = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        # Statistics
        mean_act = float(np.mean(heatmap))
        max_act = float(np.max(heatmap))
        
        # Normalize heatmap
        heatmap_min, heatmap_max = heatmap_upsampled.min(), heatmap_upsampled.max()
        if heatmap_max > heatmap_min:
            heatmap_norm = (heatmap_upsampled - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap_norm = np.zeros_like(heatmap_upsampled)
        
        # Create binary mask/contour
        threshold = args.mask_threshold * heatmap_max if heatmap_max > 0 else 0.5
        binary_mask = heatmap_upsampled > threshold
        
        results[f'concept_{neuron_idx}'] = {
            'mean_activation': mean_act,
            'max_activation': max_act,
        }
        
        col = col_idx + 1
        
        # Row 1: Heatmap only
        im = axes[0, col].imshow(heatmap_norm, cmap='viridis')
        axes[0, col].set_title(f'Neuron {neuron_idx}\nMean: {mean_act:.3f}, Max: {max_act:.3f}', fontsize=9)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        
        # Row 2: Overlay on image
        axes[1, col].imshow(img_np)
        axes[1, col].imshow(heatmap_norm, cmap='hot', alpha=0.5)
        axes[1, col].set_title(f'Overlay {neuron_idx}', fontsize=9)
        axes[1, col].axis('off')
        
        # Row 3: Contours only
        axes[2, col].imshow(img_np)
        if binary_mask.sum() > 0:
            contours = measure.find_contours(binary_mask, 0.5)
            for contour in contours:
                axes[2, col].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        axes[2, col].set_title(f'Contours {neuron_idx}', fontsize=9)
        axes[2, col].axis('off')
    
    plt.suptitle(f"Image: {image_id} | Class: {class_name} | Mask Threshold: {args.mask_threshold}", 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save combined figure with unique filename
    combined_path = save_dir / f'{image_id}_concepts.png'
    plt.savefig(combined_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return results


def main():
    args = get_args()
    
    print("="*80)
    print("HAM10000 SAE Concept Visualization (Mammo-SAE Style)")
    print("="*80)
    
    # Load top concepts JSON
    print(f"\nLoading top concepts from: {args.top_concepts_json}")
    with open(args.top_concepts_json, 'r') as f:
        top_concepts_data = json.load(f)
    
    # Extract top-k concept indices
    if isinstance(top_concepts_data, list):
        # Format: [{'rank': 0, 'index': 123, ...}, ...]
        latent_neurons = [item['index'] for item in top_concepts_data[:args.top_k]]
    elif isinstance(top_concepts_data, dict) and 'concepts' in top_concepts_data:
        # Format: {'concepts': [...]}
        latent_neurons = [item['index'] for item in top_concepts_data['concepts'][:args.top_k]]
    else:
        raise ValueError(f"Unexpected JSON format in {args.top_concepts_json}")
    
    print(f"Visualizing top {len(latent_neurons)} concepts: {latent_neurons}")
    
    # Load normalization stats
    print("\nLoading normalization stats...")
    norm_stats = torch.load(args.normalization_stats, map_location='cpu')
    feature_mean = norm_stats['mean']  # [1, 768]
    feature_std = norm_stats['std']    # [1, 768]
    print(f"✓ Loaded normalization stats from training")
    print(f"  Mean shape: {feature_mean.shape}, Std shape: {feature_std.shape}")
    
    # Load models
    print("\nLoading models...")
    panderm_model, _, preprocess = create_model_and_transforms(
        args.panderm_model,
        pretrained='default',
        device=args.device
    )
    panderm_model.eval()
    
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
    
    print(f"✓ Models loaded")
    
    # Setup feature extractor
    feature_extractor = FeatureExtractor(panderm_model)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = HAM10000_Dataset(
        args.data_split_file,
        args.image_root_dir,
        args.image_ext,
        preprocess,
        class_filter=args.class_filter
    )
    
    # Limit to num_images
    if len(dataset) > args.num_images:
        indices = torch.randperm(len(dataset))[:args.num_images].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"Visualizing {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Process images
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (pixel_values, labels, img_paths, image_ids) in enumerate(
            tqdm(dataloader, desc="Visualizing images")
        ):
            pixel_values = pixel_values.to(args.device)
            
            # Get patch features [1, 197, 768]
            _ = panderm_model.visual(pixel_values)
            patch_features = feature_extractor.get_features()
            
            # Remove CLS token [1, 196, 768]
            patch_features = patch_features[:, 1:, :]
            
            # Reshape to spatial [1, 14, 14, 768]
            B, N_patches, D = patch_features.shape
            H_patches = W_patches = 14
            patch_features = patch_features.reshape(B, H_patches, W_patches, D)
            
            # Flatten for SAE [196, 768]
            patch_features_flat = patch_features.reshape(-1, D)
            
            # DEBUG: Check raw features
            if batch_idx == 0:
                print(f"\n=== DEBUG INFO (First Image) ===")
                print(f"Raw features shape: {patch_features_flat.shape}")
                print(f"Raw features - Mean: {patch_features_flat.mean().item():.6f}, Std: {patch_features_flat.std().item():.6f}")
                print(f"Raw features - Min: {patch_features_flat.min().item():.6f}, Max: {patch_features_flat.max().item():.6f}")
                print(f"\nNormalization stats:")
                print(f"  Mean: {feature_mean.mean().item():.6f}, Std mean: {feature_std.mean().item():.6f}")
                print(f"  Mean range: [{feature_mean.min().item():.6f}, {feature_mean.max().item():.6f}]")
                print(f"  Std range: [{feature_std.min().item():.6f}, {feature_std.max().item():.6f}]")
            
            # CRITICAL: Normalize features using training statistics
            patch_features_normalized = (patch_features_flat - feature_mean.to(args.device)) / (feature_std.to(args.device) + 1e-8)
            
            # DEBUG: Check normalized features
            if batch_idx == 0:
                print(f"\nNormalized features - Mean: {patch_features_normalized.mean().item():.6f}, Std: {patch_features_normalized.std().item():.6f}")
                print(f"Normalized features - Min: {patch_features_normalized.min().item():.6f}, Max: {patch_features_normalized.max().item():.6f}")
            
            if args.n_components > 0:
                patch_features_normalized = patch_features_normalized.unsqueeze(1)
            
            # Pass through SAE (now with normalized features!)
            sae_out = sae(patch_features_normalized)
            concepts = sae_out.learned_activations
            
            # DEBUG: Check SAE output
            if batch_idx == 0:
                print(f"\nSAE output:")
                print(f"  Learned activations shape: {concepts.shape}")
                print(f"  Learned activations - Mean: {concepts.mean().item():.6f}, Std: {concepts.std().item():.6f}")
                print(f"  Learned activations - Min: {concepts.min().item():.6f}, Max: {concepts.max().item():.6f}")
                print(f"  Non-zero activations: {(concepts > 0).sum().item()} / {concepts.numel()}")
                print(f"  Sparsity (L0): {(concepts > 0).float().sum(dim=-1).mean().item():.1f}")
            
            if concepts.dim() == 3:
                concepts = concepts.squeeze(1)
            
            # Reshape to spatial [14, 14, n_learned_features]
            concepts_spatial = concepts.reshape(H_patches, W_patches, -1)
            
            # DEBUG: Check spatial concepts
            if batch_idx == 0:
                print(f"\nConcepts spatial shape: {concepts_spatial.shape}")
                print(f"  Selected neuron {latent_neurons[0]} - Mean: {concepts_spatial[:, :, latent_neurons[0]].mean().item():.6f}, Max: {concepts_spatial[:, :, latent_neurons[0]].max().item():.6f}")
                print(f"===================================\n")
            
            # Visualize
            results = visualize_topk_concepts(
                pixel_values,
                concepts_spatial,
                labels[0].item(),
                img_paths[0],
                image_ids[0],
                latent_neurons,
                args
            )
            
            all_results.append({
                'image_id': image_ids[0],
                'label': labels[0].item(),
                'concepts': results
            })
    
    # Save aggregated results per class
    # Organize results by class
    results_by_class = {}
    for result in all_results:
        class_label = result['label']
        class_name = CLASS_NAMES[class_label]
        class_name_clean = class_name.replace(" ", "_").replace("-", "_")
        
        if class_name_clean not in results_by_class:
            results_by_class[class_name_clean] = []
        results_by_class[class_name_clean].append(result)
    
    # Save separate JSON for each class
    for class_name_clean, class_results in results_by_class.items():
        class_save_dir = Path(args.save_dir).parent / f"visualization_{class_name_clean}"
        results_path = class_save_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(class_results, f, indent=2)
        print(f"✓ Saved {len(class_results)} results to {results_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"✓ Results saved to class-specific folders")
    print(f"{'='*80}")
    
    feature_extractor.close()


if __name__ == "__main__":
    main()
