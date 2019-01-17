#!/bin/bash
# First load the cuda modules so we can use the GPU
module load nvidia/cuda-9.0
module load nvidia/cuda-9.0_cudnn-7.3

projectfolder=/deepstore/datasets/course/aml/group4
source $projectfolder/env/bin/activate
python $projectfolder/src/Train_cluster_segmentation.py
deactivate
