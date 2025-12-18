# Discover-then-Name: Concept Discovery Pipeline

This document describes how to run the full concept discovery pipeline for the HAM10000 skin lesion dataset using Sparse Autoencoders (SAE) on HPC with SLURM batch jobs.

---

## 📋 Pipeline Overview

| Step | Batch Job | Description | Output |
|------|-----------|-------------|--------|
| 1 | Local (no GPU needed) | Prepare vocabulary | `master_vocabulary.txt` |
| 2 | `02_prepare_features_job.sbatch` | Extract image features | `*_patch_features_{train,val}.pt` |
| 3 | `03_train_sae_job.sbatch` | Train Sparse Autoencoder | `sparse_autoencoder_final.pt` |
| 4 | `04_find_top_concepts_per_class_job.sbatch` | Analyze top concepts per class | JSON concept rankings |
| 5 | `05_visualize_ham10000_sae_job.sbatch` | Visualize discovered concepts | PNG heatmaps |

---

## 🔧 Prerequisites

### 1. Build the Apptainer Container

```bash
apptainer build dependencies/dn_cbm_gpu_env.sif dependencies/dn_cbm_gpu_env.def
```

### 2. Prepare Data

Ensure HAM10000 images are accessible on the cluster:
```
/scratch/user/$USER/HAM10000/HAM10000_images/
├── ISIC_0024306.jpg
├── ISIC_0024307.jpg
└── ... (all .jpg images)
```

### 3. Create Required Directories

```bash
mkdir -p logs
mkdir -p results
```

---

## 🚀 Running the Pipeline (SLURM Batch Jobs)

### Quick Start

```bash
# Step 1: Prepare Vocabulary (run locally, no GPU needed)
cd new_implementation && python 01_prepare_vocabulary.py && cd ..

# Step 2: Submit feature extraction job
sbatch sh_scripts/02_prepare_features_job.sbatch
# Wait for completion...

# Step 3: Submit SAE training job
sbatch sh_scripts/03_train_sae_job.sbatch
# Wait for completion...

# Step 4: Submit concept analysis job
sbatch sh_scripts/04_find_top_concepts_per_class_job.sbatch
# Wait for completion...

# Step 5: Submit visualization job
sbatch sh_scripts/05_visualize_ham10000_sae_job.sbatch
```

### Monitor Jobs

```bash
# Check job queue
squeue -u $USER

# Watch job status
watch -n 5 squeue -u $USER

# View real-time logs
tail -f logs/step3_train_sae*.out

# Check job history
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

---

## 📝 Batch Job Details

### Step 1: Prepare Vocabulary (Local)

No SLURM job needed. Run directly:

```bash
cd new_implementation
python 01_prepare_vocabulary.py
cd ..
```

**Output:** `new_implementation/master_vocabulary.txt`

---

### Step 2: Extract Features

**Submit:**
```bash
sbatch sh_scripts/02_prepare_features_job.sbatch
```

**Job Configuration:**
```bash
#SBATCH --job-name=step2_features_embedding_dumping
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/step2_features_embedding_dumping%j.out
```

**Key Parameters (edit `02_prepare_features.sh`):**
```bash
IMAGE_ROOT_DIR="/scratch/user/uqssalem/HAM10000/HAM10000_images"
MODEL_CHOICE="dermlip"   # or "clip"
SCRATCH_DIR="/scratch/user/uqssalem/Discover_then_name"
```

**Output:**
- `$SCRATCH_DIR/features/dermlip_patch_features_train.pt`
- `$SCRATCH_DIR/features/dermlip_patch_features_val.pt`
- `./dermlip_text_embeddings.pt`

**Logs:** `logs/step2_features_embedding_dumping<JOB_ID>.out`

---

### Step 3: Train SAE

**Submit:**
```bash
sbatch sh_scripts/03_train_sae_job.sbatch
```

**Job Configuration:**
```bash
#SBATCH --job-name=step3_train_sae
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/step3_train_sae%j.out
```

**Key Parameters (edit `03_train_sae.sh`):**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--input_dim` | 768 | Feature dimension (DermLIP/CLIP) |
| `--expansion_factor` | 8 | Hidden = 768 × 8 = 6144 neurons |
| `--num_epochs` | 200 | Training epochs |
| `--batch_size` | 4096 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--l1_coeff` | 3e-5 | Sparsity coefficient |

**Output:**
- `$SCRATCH_DIR/sae_checkpoints/sparse_autoencoder_final.pt`
- Intermediate checkpoints every 10 epochs

**Logs:** `logs/step3_train_sae<JOB_ID>.out`

---

### Step 4: Find Top Concepts

**Submit:**
```bash
sbatch sh_scripts/04_find_top_concepts_per_class_job.sbatch
```

**Job Configuration:**
```bash
#SBATCH --job-name=step4_find_top_concepts
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/step4_find_top_concepts%j.out
```

**Key Parameters (edit `04_find_top_concepts_per_class.sh`):**
```bash
SAE_CHECKPOINT="$SCRATCH_DIR/sae_checkpoints/sparse_autoencoder_final.pt"
VAL_FEATURES="$SCRATCH_DIR/features/dermlip_patch_features_val.pt"
INPUT_DIM=768
EXPANSION_FACTOR=8
```

**Output:**
- `./results/sae_concepts_ham10000_val_global.pth`
- `./results/class_wise_concepts/val/top_k_concepts_class=0.json`
- `./results/class_wise_concepts/val/top_k_concepts_class=1.json`
- ... (one JSON per class)

**Logs:** `logs/step4_find_top_concepts<JOB_ID>.out`

---

### Step 5: Visualize Concepts

**Submit:**
```bash
sbatch sh_scripts/05_visualize_ham10000_sae_job.sbatch
```

**Job Configuration:**
```bash
#SBATCH --job-name=step5_visualize_concepts
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/step5_visualize_concepts%j.out
```

**Key Parameters (edit `05_visualize_ham10000_sae.sh`):**
```bash
VIS_CLASS_ID=4                    # Class to visualize (4 = Melanoma)
TOP_K=10                          # Number of top concepts
NUM_IMAGES=20                     # Images per concept
IMAGE_ROOT_DIR="/scratch/user/uqssalem/HAM10000/HAM10000_images"
```

**Output:** `./results/visualization_class_4/` (PNG heatmaps)

**Logs:** `logs/step5_visualize_concepts<JOB_ID>.out`

---

## 📊 SLURM Resources Summary

| Job | Memory | GPU | Time | Dependencies |
|-----|--------|-----|------|--------------|
| Step 2 | 32G | A100 MIG (10GB) | 12h | None |
| Step 3 | 32G | A100 MIG (10GB) | 12h | Step 2 |
| Step 4 | 8G | A100 MIG (10GB) | 12h | Step 2, 3 |
| Step 5 | 8G | A100 MIG (10GB) | 1h | Step 3, 4 |

---

## 🔗 Job Dependencies (Advanced)

Submit all jobs with dependencies automatically:

```bash
# Step 1: Run locally first
cd new_implementation && python 01_prepare_vocabulary.py && cd ..

# Step 2: Submit feature extraction
JOB2=$(sbatch --parsable sh_scripts/02_prepare_features_job.sbatch)
echo "Submitted Step 2: $JOB2"

# Step 3: Submit SAE training (depends on Step 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 sh_scripts/03_train_sae_job.sbatch)
echo "Submitted Step 3: $JOB3 (after $JOB2)"

# Step 4: Submit concept analysis (depends on Step 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 sh_scripts/04_find_top_concepts_per_class_job.sbatch)
echo "Submitted Step 4: $JOB4 (after $JOB3)"

# Step 5: Submit visualization (depends on Step 4)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 sh_scripts/05_visualize_ham10000_sae_job.sbatch)
echo "Submitted Step 5: $JOB5 (after $JOB4)"

echo ""
echo "Pipeline submitted! Monitor with: squeue -u $USER"
```

---

## 🏷️ HAM10000 Class Labels

| Class ID | Diagnosis | Full Name |
|----------|-----------|-----------|
| 0 | akiec | Actinic Keratoses |
| 1 | bcc | Basal Cell Carcinoma |
| 2 | bkl | Benign Keratosis-like Lesions |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic Nevi |
| 6 | vasc | Vascular Lesions |

---

## ⚙️ Configuration Paths

Edit these paths in the shell scripts before running:

| Variable | Location | Description |
|----------|----------|-------------|
| `IMAGE_ROOT_DIR` | `02_prepare_features.sh`, `05_visualize*.sh` | HAM10000 images folder |
| `SCRATCH_DIR` | All scripts | Directory for large files |
| `SAE_CHECKPOINT` | `04_*.sh`, `05_*.sh` | Trained SAE model path |

**Default SCRATCH_DIR:**
```bash
/scratch/user/uqssalem/Discover_then_name/
├── features/
│   ├── dermlip_patch_features_train.pt
│   └── dermlip_patch_features_val.pt
└── sae_checkpoints/
    └── sparse_autoencoder_final.pt
```

---

## 🐛 Troubleshooting

### Job Failed

```bash
# Check exit code
sacct -j <JOB_ID> --format=JobID,State,ExitCode

# View full log
cat logs/step*<JOB_ID>.out
```

### Common Issues

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` in shell script |
| `File not found` | Check paths in shell scripts; verify previous step completed |
| `SAE checkpoint not found` | Ensure Step 3 completed; check `$SCRATCH_DIR/sae_checkpoints/` |
| `Container not found` | Run `apptainer build` first |

### Verify Outputs

```bash
# Check Step 2 features
python -c "import torch; d = torch.load('/scratch/user/$USER/Discover_then_name/features/dermlip_patch_features_val.pt'); print(f'Samples: {len(d)}')"

# Check Step 3 checkpoint exists
ls -la /scratch/user/$USER/Discover_then_name/sae_checkpoints/

# View Step 4 top concepts
cat results/class_wise_concepts/val/top_k_concepts_class=4.json | python -m json.tool | head -30


