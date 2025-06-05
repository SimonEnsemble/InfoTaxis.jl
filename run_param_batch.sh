#!/bin/bash
#SBATCH -J param_test_batch
#SBATCH -A mySponsoredAccount
#SBATCH -p share
#SBATCH -o slurm_out/param_batch_%A_%a.out
#SBATCH -e slurm_out/param_batch_%A_%a.err
#SBATCH --array=1-5             #num of jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00

module load julia/1.9.4

#NOTE: this runs the run_test_param_subset.jl file intended solely for SLURM tasks!!!
julia --threads 4 scripts/run_test_param_subset.jl $SLURM_ARRAY_TASK_ID
