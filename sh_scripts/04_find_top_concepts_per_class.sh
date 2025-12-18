#!/bin/bash
set -e

echo "=========================================="
echo "Step 4A: Analyze SAE Concepts (CORRECTED)"
echo "=========================================="
echo ""
echo "This script uses EXISTING patch features"
echo "from Step 2 (no re-extraction needed!)"
echo ""

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================

# --- INPUT: Patch features from Step 2 ---
# These are the files you already generated!
MODEL_CHOICE="dermlip"
SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
TRAIN_FEATURES="$SCRATCH_DIR/features/${MODEL_CHOICE}_patch_features_train.pt"
VAL_FEATURES="$SCRATCH_DIR/features/${MODEL_CHOICE}_patch_features_val.pt"

# --- INPUT: Trained SAE from Step 3 ---
SAE_CHECKPOINT="$SCRATCH_DIR/sae_checkpoints/sparse_autoencoder_final.pt"

# --- SAE Architecture (MUST MATCH Step 3 training) ---
INPUT_DIM=768
EXPANSION_FACTOR=8  # Change to 16 if you used expansion_factor=16 in training
N_COMPONENTS=1

# --- Output ---
RESULTS_DIR="./results"

# --- Processing ---
BATCH_SIZE=128  # Batch size for processing images (can be higher than training)
DEVICE="cuda"   # or "cpu"

# ============================================
# VERIFY FILES EXIST
# ============================================

echo "Verifying input files..."
echo ""

if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "❌ ERROR: SAE checkpoint not found: $SAE_CHECKPOINT"
    echo "   Make sure you've run Step 3 (train SAE) first!"
    exit 1
fi
echo "✓ SAE checkpoint found: $SAE_CHECKPOINT"

if [ ! -f "$TRAIN_FEATURES" ]; then
    echo "❌ ERROR: Train features not found: $TRAIN_FEATURES"
    echo "   Make sure you've run Step 2 (extract features) first!"
    exit 1
fi
echo "✓ Train features found: $TRAIN_FEATURES"

if [ ! -f "$VAL_FEATURES" ]; then
    echo "❌ ERROR: Val features not found: $VAL_FEATURES"
    echo "   Make sure you've run Step 2 (extract features) first!"
    exit 1
fi
echo "✓ Val features found: $VAL_FEATURES"

if [ ! -f "new_implementation/04_find_top_concepts_per_class.py" ]; then
    echo "❌ ERROR: 04_find_top_concepts_per_class.py not found"
    echo "   Make sure the script is in the current directory!"
    exit 1
fi
echo "✓ Analysis script found"

echo ""
echo "All checks passed!"
echo ""

# ============================================
# ANALYZE VALIDATION SET
# ============================================

echo "=========================================="
echo "Analyzing Validation Set"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Load patch features from Step 2"
echo "  2. Pass through trained SAE"
echo "  3. Spatially average concepts"
echo "  4. Rank by class importance"
echo ""
echo "Expected time: ~2-5 minutes on GPU"
echo ""

python new_implementation/04_find_top_concepts_per_class.py \
    --patch_features_file "$VAL_FEATURES" \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --input_dim "$INPUT_DIM" \
    --expansion_factor "$EXPANSION_FACTOR" \
    --n_components "$N_COMPONENTS" \
    --batch_size "$BATCH_SIZE" \
    --split_name "val" \
    --save_dir "$RESULTS_DIR" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Validation analysis failed"
    exit 1
fi

echo ""
echo "✓✓✓ Validation analysis complete!"
echo ""

# ============================================
# ANALYZE TRAINING SET (OPTIONAL)
# ============================================

read -p "Do you want to analyze the training set too? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=========================================="
    echo "Analyzing Training Set"
    echo "=========================================="
    echo ""
    
    python find_top_concepts_per_class_CORRECTED.py \
        --patch_features_file "$TRAIN_FEATURES" \
        --sae_checkpoint "$SAE_CHECKPOINT" \
        --input_dim "$INPUT_DIM" \
        --expansion_factor "$EXPANSION_FACTOR" \
        --n_components "$N_COMPONENTS" \
        --batch_size "$BATCH_SIZE" \
        --split_name "train" \
        --save_dir "$RESULTS_DIR" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Training analysis failed"
        exit 1
    fi
    
    echo ""
    echo "✓✓✓ Training analysis complete!"
    echo ""
fi

# ============================================
# SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "🎉 Step 4A Complete! 🎉"
echo "=========================================="
echo ""
echo "📂 Results saved to: $RESULTS_DIR"
echo ""
echo "Outputs:"
echo "  1. Global concepts (.pth files):"
echo "     - $RESULTS_DIR/sae_concepts_ham10000_val_global.pth"
if [ -f "$RESULTS_DIR/sae_concepts_ham10000_train_global.pth" ]; then
    echo "     - $RESULTS_DIR/sae_concepts_ham10000_train_global.pth"
fi
echo ""
echo "  2. Class rankings (JSON files):"
echo "     - $RESULTS_DIR/class_wise_concepts/val/"
if [ -d "$RESULTS_DIR/class_wise_concepts/train/" ]; then
    echo "     - $RESULTS_DIR/class_wise_concepts/train/"
fi
echo ""
echo "=========================================="
echo "📊 Quick Analysis"
echo "=========================================="
echo ""

# Show top-5 concepts for Melanoma (class 4)
if [ -f "$RESULTS_DIR/class_wise_concepts/val/top_k_concepts_class=4.json" ]; then
    echo "Top-5 concepts for Melanoma (Class 4):"
    python << 'EOF'
import json
with open('./results/class_wise_concepts/val/top_k_concepts_class=4.json') as f:
    concepts = json.load(f)
for i, c in enumerate(concepts[:5]):
    print(f"  {i+1}. Concept {c['index']:4d}: importance={c['importance']:.4f}")
EOF
    echo ""
fi

echo "=========================================="
echo "📝 Next Steps"
echo "=========================================="
echo ""
echo "View all concept rankings:"
echo "  cat $RESULTS_DIR/class_wise_concepts/val/top_k_concepts_class=4.json | head -30"
echo ""
echo "Compare classes:"
echo "  for i in {0..6}; do"
echo "    echo \"Class \$i:\""
echo "    cat $RESULTS_DIR/class_wise_concepts/val/top_k_concepts_class=\$i.json | head -6 | tail -3"
echo "  done"
echo ""
echo "Run Step 4B (visualization) next!"
echo ""
echo "=========================================="