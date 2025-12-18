#!/bin/bash
#SBATCH --job-name=step5_visualize_concepts
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/step5_visualize_concepts%j.out

echo "========================================="
echo "step3_visualize_concepts - GPU Job (MIG)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================="

# Load modules (if needed)
# module load anaconda3/2021.11
# module load cuda/11.6
# conda activate my_env

set -e
echo "--- 5/6: Visualization of SAE concepts ---"


# Run the python script
apptainer exec --nv \
    --writable-tmpfs \
    --bind $(pwd):/app \
    dependencies/dn_cbm_gpu_env.sif bash sh_scripts/05_visualize_ham10000_sae.sh

echo "--- Step 3 Complete ---"
