#!/bin/bash
# THIS IS AN EXAMPLE - YOU NEED TO FILL IN THE PATHS

# --- Step 5: Visualization ---
echo "=========================================="
echo "Step 5: Visualize SAE Concepts"
echo "=========================================="

SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
RESULTS_DIR="./results"

# --- INPUT: Trained SAE ---
SAE_CHECKPOINT="$SCRATCH_DIR/sae_checkpoints/sparse_autoencoder_final.pt"

# --- INPUT: Top concepts JSON from Step 4A ---
# Visualize concepts for Melanoma (Class 4)
TOP_CONCEPTS_JSON="$RESULTS_DIR/class_wise_concepts/val/top_k_concepts_class=4.json"
VIS_CLASS_ID=4 

# --- INPUT: HAM10000 Data ---
# Path to the pickle file (e.g., class2images_val.p)
DATA_SPLIT_FILE="new_implementation/dataset/splits/class2images_val.p"
# TRAIN_SPLIT_FILE="new_implementation/dataset/splits/class2images_train.p"
# VAL_SPLIT_FILE="new_implementation/dataset/splits/class2images_val.p"
# Path to the root folder of HAM10000 images (e.g., /data/ham10000/images/)
IMAGE_ROOT_DIR="/scratch/user/uqssalem/HAM10000/HAM10000_images"

# --- Output ---
SAVE_DIR="$RESULTS_DIR/visualization_class_${VIS_CLASS_ID}"

python new_implementation/05_visualize_ham10000_sae.py \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --data_split_file "$DATA_SPLIT_FILE" \
    --image_root_dir "$IMAGE_ROOT_DIR" \
    --top_concepts_json "$TOP_CONCEPTS_JSON" \
    --class_filter "$VIS_CLASS_ID" \
    --save_dir "$SAVE_DIR" \
    --input_dim 768 \
    --expansion_factor 8 \
    --top_k 10 \
    --num_images 20 \
    --device cuda

echo "Visualization complete! Check results in $SAVE_DIR"