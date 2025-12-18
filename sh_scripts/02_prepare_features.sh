#!/bin/bash
set -e
echo "--- 2/6: Extracting Features & Embeddings ---"

# --- CONFIGURATION: SET YOUR PATHS HERE ---
# Path to the split files (both train and val will be extracted)
TRAIN_SPLIT_FILE="new_implementation/dataset/splits/class2images_train.p"
VAL_SPLIT_FILE="new_implementation/dataset/splits/class2images_val.p"
# Path to the folder containing all .jpg images
IMAGE_ROOT_DIR="/scratch/user/uqssalem/HAM10000/HAM10000_images"
# Model choice: 'dermlip' or 'clip'
MODEL_CHOICE="dermlip"
# -------------------------------------------

# Use scratch directory for large feature files
SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
mkdir -p "$SCRATCH_DIR/features"

# Define output filenames based on model choice
TEXT_EMBED_OUTPUT="./${MODEL_CHOICE}_text_embeddings.pt"  # Small, keep in current dir
TRAIN_FEATURE_OUTPUT="$SCRATCH_DIR/features/${MODEL_CHOICE}_patch_features_train.pt"
VAL_FEATURE_OUTPUT="$SCRATCH_DIR/features/${MODEL_CHOICE}_patch_features_val.pt"

# Run the python script (extracts from both train and val splits separately)
python new_implementation/02_prepare_features_and_embeddings.py \
    --train_split_file "$TRAIN_SPLIT_FILE" \
    --val_split_file "$VAL_SPLIT_FILE" \
    --train_feature_output "$TRAIN_FEATURE_OUTPUT" \
    --val_feature_output "$VAL_FEATURE_OUTPUT" \
    --image_root_dir "$IMAGE_ROOT_DIR" \
    --image_ext ".jpg" \
    --model_choice "$MODEL_CHOICE" \
    --text_embed_output "$TEXT_EMBED_OUTPUT" \
    --device "cpu" # Use "cuda" if you have a local GPU

# apptainer exec --nv --writable-tmpfs --bind $(pwd):/app concept_discovery/depnd/concept_dis_gpu.sif bash your_script.sh

echo "--- Step 2 Complete ---"
