#!/bin/bash
#SBATCH --job-name=step4_find_top_concepts
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/step4_find_top_concepts%j.out

echo "========================================="
echo "step3_find_top_concepts - GPU Job (MIG)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================="

# Load modules (if needed)
# module load anaconda3/2021.11
# module load cuda/11.6
# conda activate my_env

set -e
echo "--- 4/6: find top concepts ---"


# Run the python script
apptainer exec --nv \
    --writable-tmpfs \
    --bind $(pwd):/app \
    dependencies/dn_cbm_gpu_env.sif bash sh_scripts/04_find_top_concepts_per_class.sh

echo "--- Step 3 Complete ---"
