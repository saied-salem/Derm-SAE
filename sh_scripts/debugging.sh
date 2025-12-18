#!/bin/bash
set -e
echo "--- 2/6: Extracting Features & Embeddings ---"

# --- CONFIGURATION: SET YOUR PATHS HERE ---
# Path to the .p file for the TRAINING split
DATA_SPLIT_FILE="new_implementation/dataset/splits/class2images_train.p"
# Path to the folder containing all .jpg images
IMAGE_ROOT_DIR="/scratch/user/uqssalem/HAM10000/HAM10000_images"
# Model choice: 'dermlip' or 'clip'
MODEL_CHOICE="dermlip"
# -------------------------------------------

# Define output filenames based on model choice
TEXT_EMBED_OUTPUT="./${MODEL_CHOICE}_text_embeddings.pt"
PATCH_FEATURE_OUTPUT="./${MODEL_CHOICE}_patch_features.pt"

# Run the python script
python new_implementation/debugging.py 

# apptainer exec --nv --writable-tmpfs --bind $(pwd):/app concept_discovery/depnd/concept_dis_gpu.sif bash your_script.sh

echo "--- Step 2 Complete ---"
