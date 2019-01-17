#!/bin/bash
#SBATCH --partition=main
#SBATCH -N 1
#SBATCH -J segmentationV1
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
./job-gpu-seg-v1.sh
