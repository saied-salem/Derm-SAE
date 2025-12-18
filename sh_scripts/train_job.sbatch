#!/bin/bash
#SBATCH --job-name=train_ada_cbm
#SBATCH --account=a_hcc
#SBATCH --partition=gpu_cuda
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_ada_cbm%j.out


echo "========================================="
echo "train_ada_cbm - GPU Job (MIG)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================="

# Show GPU info #SBATCH --error=logs/train_ada_cbm%j.err
# nvidia-smi

#chmod +x downstream_preparation/scripts/train_ada_cbm.sh


# Run with GPU support (--nv flag is critical!)
apptainer exec --nv \
    --writable-tmpfs \
    --bind $(pwd):/app \
    dependencies/ada_cbm_gpu_env.sif bash train_simple.sh 
echo "Finished: $(date) | Exit: $?"
