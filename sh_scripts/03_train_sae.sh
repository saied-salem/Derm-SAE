#!/bin/bash

# Step 3: Train Sparse Autoencoder on HAM10000 Patch Features
# Using scratch directory for large outputs

SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
mkdir -p "$SCRATCH_DIR/sae_training_output"
mkdir -p "$SCRATCH_DIR/sae_checkpoints"

python new_implementation/03_train_sae.py \
    --train_feature_file "$SCRATCH_DIR/features/dermlip_patch_features_train.pt" \
    --val_feature_file "$SCRATCH_DIR/features/dermlip_patch_features_val.pt" \
    --output_dir "$SCRATCH_DIR/sae_training_output" \
    --checkpoint_dir "$SCRATCH_DIR/sae_checkpoints" \
    --input_dim 768 \
    --expansion_factor 8 \
    --num_epochs 200 \
    --batch_size 4096 \
    --lr 3e-4 \
    --l1_coeff 3e-5 \
    --checkpoint_freq 10 \
    --val_freq 1 \
    --resample_freq 10 \
    --use_wandb \
    --wandb_project "HAM10000-SAE" \
    --experiment_name "panderm_sae_exp1" \
    --device cuda

echo "Training complete!"

# apptainer exec --nv --writable-tmpfs --bind $(pwd):/app concept_discovery/depnd/concept_dis_gpu.sif bash your_script.sh

echo "--- Step 3 Complete ---"
