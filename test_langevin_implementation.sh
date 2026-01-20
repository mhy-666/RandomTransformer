#!/bin/bash
#SBATCH --job-name=test_langevin_implementation
#SBATCH --output=${job_name}_%j.out
#SBATCH --error=${job_name}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-common

# ImageNet 数据路径
DATA_DIR="/work/hm235/random_transformer/data/hf_cache/"

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3" 
CONDA_ENV="tinyvit"
# 初始化conda
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

pytest test_langevin_implementation.py -v --tb=short