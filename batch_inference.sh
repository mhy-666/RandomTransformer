#!/bin/bash

# ============================================================================
# Batch Inference Script - 评估所有final_model
# ============================================================================
#SBATCH --job-name=batch_inference
#SBATCH --output=logs/batch_inference_svd_continual_learning.out
#SBATCH --error=logs/batch_inference_svd_continual_learning.err
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

# 创建日志目录
mkdir -p logs

# 基础配置
exp="svd_continual_learning"
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# 实验目录配置
BASE_DIR="/work/hm235/random_transformer/outputs/${exp}"
OUTPUT_FILE="all_results.csv"

# 激活conda环境
cd ${WORK_DIR}
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

# 设置环境变量
export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "Batch Inference - Evaluating All Final Models"
echo "========================================================================"
echo "Base directory: ${BASE_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Start time: $(date)"
echo "========================================================================"

# 运行batch inference
python batch_inference.py \
    --base_dir ${BASE_DIR} \
    --output_file ${OUTPUT_FILE} \
    --device cuda \
    --max_length 1024

echo "========================================================================"
echo "Batch inference completed!"
echo "End time: $(date)"
echo "Results saved to: ${BASE_DIR}/${OUTPUT_FILE}"
echo "========================================================================"
