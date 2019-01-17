#!/bin/bash
#SBATCH --partition=main
#SBATCH -N 1
#SBATCH -J drrV2
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
./job-gpu-v2.sh
