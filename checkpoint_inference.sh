#!/bin/bash

# ============================================================================
# Checkpoint Evolution Inference - 单个实验的所有checkpoint评估
# ============================================================================

#SBATCH --job-name=ckpt_inference
#SBATCH --output=logs/ckpt_inference_%j.out
#SBATCH --error=logs/ckpt_inference_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-common

# 创建日志目录
mkdir -p logs

# 基础配置
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# 实验目录配置（根据需要修改）
EXP_DIR="/work/hm235/random_transformer/outputs/freeze_random_bp/freeze_attn_rbp_attn_fixed_seed42"
TOTAL_STEPS=80000  # 指定总训练步数
OUTPUT_NAME="checkpoint_evolution"

# 可选：只评估特定范围的checkpoint
START_STEP=0
END_STEP=999999

# 激活conda环境
cd ${WORK_DIR}
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

# 设置环境变量
export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "Checkpoint Evolution Inference"
echo "========================================================================"
echo "Experiment directory: ${EXP_DIR}"
echo "Total training steps: ${TOTAL_STEPS}"
echo "Output name: ${OUTPUT_NAME}"
echo "Checkpoint range: ${START_STEP} - ${END_STEP}"
echo "Start time: $(date)"
echo "========================================================================"

# 运行checkpoint inference
python checkpoint_inference.py \
    --exp_dir ${EXP_DIR} \
    --total_steps ${TOTAL_STEPS} \
    --output_name ${OUTPUT_NAME} \
    --device cuda \
    --eval_nq_samples 1000 \
    --eval_lambada_samples 1000 \
    --eval_wikitext_samples 1000 \
    --max_length 1024 \
    --start_step ${START_STEP} \
    --end_step ${END_STEP}

echo "========================================================================"
echo "Checkpoint inference completed!"
echo "End time: $(date)"
echo "Results saved to: ${EXP_DIR}/${OUTPUT_NAME}.csv"
echo "Plot saved to: ${EXP_DIR}/${OUTPUT_NAME}.png"
echo "========================================================================"
