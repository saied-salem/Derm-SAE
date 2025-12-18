# HAM10000-SAE: Complete Usage Guide

Following the Mammo-SAE methodology for skin lesion analysis.

## Training Results Assessment

Your training results show:
- **~3,140-4,010 alive neurons** out of 6,144 (768 × 8)
- **~1,500 L0 sparsity** (average active features per sample)
- **~2,879-3,004 "almost dead" neurons**

### Recommendations for Better Training

```bash
# Try these improved hyperparameters:
python 03_train_sae.py \
    --train_feature_file ./dermlip_patch_features_train.pt \
    --val_feature_file ./dermlip_patch_features_val.pt \
    --output_dir ./sae_training_output \
    --checkpoint_dir ./sae_checkpoints \
    --input_dim 768 \
    --expansion_factor 16 \  # Increased from 8
    --l1_coeff 1e-4 \        # Increased from 3e-5
    --lr 3e-4 \
    --batch_size 4096 \
    --num_epochs 200 \
    --resample_freq 5 \      # More frequent resampling
    --use_wandb
```

Key changes:
1. **Expansion factor 16**: More capacity (12,288 features)
2. **Higher L1 coefficient**: Stronger sparsity penalty
3. **More frequent resampling**: Better handling of dead neurons

---

## Complete Pipeline (Mammo-SAE Style)

### Step 1: Extract Global Concept Strengths

This extracts SAE concepts for all images and averages them spatially (following `save_concept_strengths_global.py`).

```bash
python find_top_concepts_per_class.py \
    --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
    --data_split_file ./class2images_val.p \
    --image_root_dir /path/to/HAM10000/images \
    --image_ext .jpg \
    --input_dim 768 \
    --expansion_factor 8 \
    --n_components 1 \
    --split_name val \
    --batch_size 32 \
    --save_dir ./results
```

**Output:**
- `./results/sae_concepts_ham10000_val_global.pth` - Contains:
  - `sae_concepts`: [N_images, 6144] averaged concept activations
  - `labels`: [N_images] class labels
  - `img_paths`: List of image paths

**Then analyzes concepts per class:**
- `./results/class_wise_concepts/val/top_k_concepts_class=0.json`
- `./results/class_wise_concepts/val/top_k_concepts_class=1.json`
- ...
- `./results/class_wise_concepts/val/top_k_concepts_class=all.json`

Each JSON contains ranked concepts:
```json
[
  {"rank": 0, "index": 1867, "importance": 0.4523},
  {"rank": 1, "index": 3492, "importance": 0.4102},
  ...
]
```

---

### Step 2: Visualize Top Concepts

Visualize the top-k concepts on actual images (following `visualize_latent_neuron.py`).

#### Example 1: Visualize Melanoma concepts

```bash
python visualize_ham10000_sae.py \
    --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
    --data_split_file ./class2images_val.p \
    --image_root_dir /path/to/HAM10000/images \
    --image_ext .jpg \
    --top_concepts_json ./results/class_wise_concepts/val/top_k_concepts_class=4.json \
    --top_k 10 \
    --num_images 20 \
    --class_filter 4 \
    --mask_threshold 0.5 \
    --split_name val \
    --save_dir ./results/visualization
```

**Output structure:**
```
results/visualization/visuals_mask_th=0.5/
├── class=4/
│   ├── image_id=ISIC_0024306/
│   │   ├── original_image.png
│   │   ├── neuron=1867_heatmap.png
│   │   ├── neuron=1867_heatmap_with_mask.png
│   │   ├── neuron=3492_heatmap.png
│   │   ├── neuron=3492_heatmap_with_mask.png
│   │   └── ...
│   ├── image_id=ISIC_0024307/
│   │   └── ...
└── all_results.json
```

#### Example 2: Visualize class-agnostic concepts

```bash
python visualize_ham10000_sae.py \
    --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
    --data_split_file ./class2images_val.p \
    --image_root_dir /path/to/HAM10000/images \
    --top_concepts_json ./results/class_wise_concepts/val/top_k_concepts_class=all.json \
    --top_k 10 \
    --num_images 50 \
    --mask_threshold 0.3 \
    --save_dir ./results/visualization_all_classes
```

#### Example 3: Visualize all 7 classes

```bash
# Loop through all classes
for class_idx in {0..6}; do
    python visualize_ham10000_sae.py \
        --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
        --data_split_file ./class2images_val.p \
        --image_root_dir /path/to/HAM10000/images \
        --top_concepts_json ./results/class_wise_concepts/val/top_k_concepts_class=${class_idx}.json \
        --top_k 10 \
        --num_images 15 \
        --class_filter ${class_idx} \
        --mask_threshold 0.5 \
        --save_dir ./results/visualization_class_${class_idx}
done
```

---

## Understanding the Outputs

### 1. Concept Rankings (JSON files)

Each class has a ranked list of concepts by importance:

```json
{
  "rank": 0,
  "index": 1867,      // Concept/neuron ID
  "importance": 0.4523  // Mean activation across all class images
}
```

**High importance = Strong association with that class**

### 2. Heatmap Visualizations

For each image and concept:

- **`neuron=X_heatmap.png`**: Shows where concept X is activated
  - Red/yellow = High activation
  - Blue/purple = Low activation

- **`neuron=X_heatmap_with_mask.png`**: Binary mask overlay
  - Red regions = Activation above threshold

### 3. Results JSON

`all_results.json` contains:
```json
[
  {
    "image_id": "ISIC_0024306",
    "label": 4,
    "concepts": {
      "concept_1867": {
        "mean_activation": 0.4523,
        "max_activation": 0.8912
      },
      ...
    }
  }
]
```

---

## Advanced Usage

### Custom Threshold Analysis

Test different mask thresholds to find optimal localization:

```bash
for threshold in 0.3 0.5 0.7; do
    python visualize_ham10000_sae.py \
        --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
        --data_split_file ./class2images_val.p \
        --image_root_dir /path/to/HAM10000/images \
        --top_concepts_json ./results/class_wise_concepts/val/top_k_concepts_class=4.json \
        --top_k 10 \
        --num_images 20 \
        --class_filter 4 \
        --mask_threshold ${threshold} \
        --save_dir ./results/viz_threshold_${threshold}
done
```

### Compare Train vs Val Concepts

```bash
# Extract train concepts
python find_top_concepts_per_class.py \
    --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
    --data_split_file ./class2images_train.p \
    --image_root_dir /path/to/HAM10000/images \
    --split_name train \
    --save_dir ./results

# Extract val concepts
python find_top_concepts_per_class.py \
    --sae_checkpoint ./sae_checkpoints/best_checkpoint.pt \
    --data_split_file ./class2images_val.p \
    --image_root_dir /path/to/HAM10000/images \
    --split_name val \
    --save_dir ./results

# Compare: Are top concepts consistent?
# Check overlap between:
# - ./results/class_wise_concepts/train/top_k_concepts_class=4.json
# - ./results/class_wise_concepts/val/top_k_concepts_class=4.json
```

---

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"

```bash
# Reduce batch size
python find_top_concepts_per_class.py \
    --batch_size 16 \  # Reduced from 32
    ...
```

### Issue: "FileNotFoundError: No such file"

Check your paths:
```bash
ls ./class2images_val.p
ls /path/to/HAM10000/images/ISIC_*.jpg
```

### Issue: Empty visualizations

- Check if concepts are actually activated
- Try lower `--mask_threshold` (e.g., 0.3 instead of 0.5)
- Verify SAE checkpoint is correct

---

## Comparison with Mammo-SAE

| Aspect | Mammo-SAE | HAM10000-SAE |
|--------|-----------|--------------|
| **Dataset** | VinDr mammograms | HAM10000 skin lesions |
| **Base model** | MammoCLIP (B5) | PanDerm (ViT-B/16) |
| **Patches** | Variable resolution | 14×14 (224×224 input) |
| **Classes** | Binary (calcification/mass) | 7 skin lesion types |
| **Ground truth** | Bounding boxes | Image-level labels only |
| **SAE input** | 512-dim features | 768-dim features |
| **Expansion** | 32x (16,384 features) | 8x or 16x recommended |

---

## Next Steps: Interventions

After visualization, you can implement interventions (like Mammo-SAE):

### Top-k Activation Intervention
Activate only top-k concepts, zero out others → measure downstream performance

### Top-k Deactivation Intervention  
Zero out top-k concepts, keep others → measure performance drop

This requires integrating a downstream classifier on HAM10000 (similar to their MammoCLIP classifier).

---

## Tips for Better Results

1. **Use higher expansion factor** (16x or 32x) for more interpretable concepts
2. **Tune L1 coefficient** to balance sparsity and reconstruction
3. **Visualize multiple thresholds** to understand concept localization
4. **Compare concepts across classes** to find discriminative features
5. **Check concept consistency** between train and val splits

---

## Citation

If you use this for research, cite both:
- Original HAM10000 dataset
- Mammo-SAE paper (methodology)
- PanDerm model (base encoder)
