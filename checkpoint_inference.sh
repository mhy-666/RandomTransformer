#!/bin/bash

# ============================================================================
# Checkpoint Evolution Inference - 单个实验的所有checkpoint评估
# ============================================================================

#SBATCH --job-name=ckpt_inference
#SBATCH --output=logs/ckpt_inference_%j.out
#SBATCH --error=logs/ckpt_inference_%j.err
#SBATCH --account=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

# 创建日志目录
mkdir -p logs

# 基础配置
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# 实验目录配置（根据需要修改）
EXP_DIR="/work/hm235/random_transformer/outputs/from_scratch_40_minibatch_size_64_grad_accum_2/baseline_1_full_finetune/baseline_1_full_finetune_seed4_qkv_all"
TOTAL_STEPS=20000  # 指定总训练步数
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
    --max_length 1024 \
    --start_step ${START_STEP} \
    --end_step ${END_STEP}

echo "========================================================================"
echo "Checkpoint inference completed!"
echo "End time: $(date)"
echo "Results saved to: ${EXP_DIR}/${OUTPUT_NAME}.csv"
echo "Plot saved to: ${EXP_DIR}/${OUTPUT_NAME}.png"
echo "========================================================================"
