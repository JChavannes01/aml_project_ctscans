#!/bin/bash
#SBATCH --partition=main
#SBATCH -N 1
#SBATCH -J basicV1
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
./job-gpu-basic-v1.sh
