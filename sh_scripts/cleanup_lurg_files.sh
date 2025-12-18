#!/bin/bash

# Cleanup script to free up disk space
# Moves large files to scratch and removes unnecessary files

SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
CURRENT_DIR=$(pwd)

echo "=========================================="
echo "Cleaning up large files to free disk space"
echo "=========================================="

# Create scratch directories
mkdir -p "$SCRATCH_DIR/features"
mkdir -p "$SCRATCH_DIR/activations"
mkdir -p "$SCRATCH_DIR/checkpoints"

# Files to move to scratch (not delete, in case we need them)
echo ""
echo "Moving large feature files to scratch..."
if [ -f "./dermlip_patch_features_train.pt" ]; then
    mv "./dermlip_patch_features_train.pt" "$SCRATCH_DIR/features/" 2>/dev/null || cp "./dermlip_patch_features_train.pt" "$SCRATCH_DIR/features/" && rm "./dermlip_patch_features_train.pt"
    echo "  ✓ Moved dermlip_patch_features_train.pt"
fi

if [ -f "./dermlip_patch_features_val.pt" ]; then
    mv "./dermlip_patch_features_val.pt" "$SCRATCH_DIR/features/" 2>/dev/null || cp "./dermlip_patch_features_val.pt" "$SCRATCH_DIR/features/" && rm "./dermlip_patch_features_val.pt"
    echo "  ✓ Moved dermlip_patch_features_val.pt"
fi

if [ -f "./dermlip_patch_features.pt" ]; then
    mv "./dermlip_patch_features.pt" "$SCRATCH_DIR/features/" 2>/dev/null || cp "./dermlip_patch_features.pt" "$SCRATCH_DIR/features/" && rm "./dermlip_patch_features.pt"
    echo "  ✓ Moved dermlip_patch_features.pt"
fi

# Delete activation files (can be regenerated)
echo ""
echo "Deleting activation files (can be regenerated)..."
if [ -d "./sae_training_output/activations" ]; then
    rm -rf "./sae_training_output/activations"
    echo "  ✓ Deleted ./sae_training_output/activations"
fi

# Move old checkpoints to scratch (keep only last 3)
echo ""
echo "Managing checkpoints..."
if [ -d "./sae_checkpoints" ]; then
    # Count checkpoints
    checkpoint_count=$(find "./sae_checkpoints" -name "*.pt" | wc -l)
    if [ "$checkpoint_count" -gt 3 ]; then
        # Move all but keep newest 3
        find "./sae_checkpoints" -name "*.pt" -type f -printf '%T@ %p\n' | sort -rn | tail -n +4 | cut -d' ' -f2- | while read file; do
            mv "$file" "$SCRATCH_DIR/checkpoints/" 2>/dev/null || true
        done
        echo "  ✓ Moved old checkpoints to scratch (kept 3 newest)"
    fi
fi

# Delete temporary files
echo ""
echo "Deleting temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null
echo "  ✓ Cleaned temp files and cache"

# Delete wandb cache (will be recreated)
echo ""
echo "Cleaning wandb cache..."
if [ -d "./sae_training_output/.wandb_cache" ]; then
    rm -rf "./sae_training_output/.wandb_cache"
    echo "  ✓ Deleted wandb cache (will be recreated)"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo ""
echo "Large files moved to: $SCRATCH_DIR"
echo ""
echo "To use moved feature files, update your scripts:"
echo "  --train_feature_file $SCRATCH_DIR/features/dermlip_patch_features_train.pt"
echo "  --val_feature_file $SCRATCH_DIR/features/dermlip_patch_features_val.pt"
echo "=========================================="

# Show disk usage
echo ""
echo "Current disk usage:"
df -h . | tail -1

