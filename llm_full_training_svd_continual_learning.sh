#!/bin/bash

# ========================================
# SVD Continual Learning 实验
# Stage 1: WikiText-103 训练 W_eff = PW + (I-P)R，每个batch重采样R
# Stage 2: TinyStories 训练，冻结PW，只训练R
# ========================================


SEEDS=(42)
MODEL_SIZE="gpt2"
BATCH_SIZE=4
GRAD_ACCUM=4

# 目录配置
BASE_DIR="/work/hm235/random_transformer"
OUTPUT_DIR="${BASE_DIR}/outputs/svd_continual_learning"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# ========== SVD 配置 ==========
SVD_RANKS=(256 512)  # 测试不同的SVD秩
APPLY_SVD_TO=("all" "attn" "mlp")  # 应用SVD到哪些层

# ========== 训练配置 ==========
# Stage 1 (WikiText-103)
STAGE1_MAX_STEPS=80000
STAGE1_LR=5e-4
STAGE1_SAVE_STEPS=2000
STAGE1_EVAL_STEPS=2000
STAGE1_LOGGING_STEPS=100

# Stage 2 (TinyStories)
STAGE2_MAX_STEPS=20000
STAGE2_LR=5e-5
STAGE2_SAVE_STEPS=2000
STAGE2_EVAL_STEPS=2000
STAGE2_LOGGING_STEPS=100

echo "========================================"
echo "SVD Continual Learning Experiments"
echo "========================================"
echo ""
echo "Experiment Design:"
echo "  Stage 1: Train on WikiText-103 with W_eff = PW + (I-P)R"
echo "           - P = UU^T from SVD of W (rank-r projection)"
echo "           - Resample R each batch"
echo "           - Train both W and R"
echo ""
echo "  Stage 2: Train on TinyStories"
echo "           - Freeze PW (preserve WikiText knowledge)"
echo "           - Train R only (learn in null space)"
echo ""
echo "SVD Configurations:"
for rank in "${SVD_RANKS[@]}"; do
    echo "  - Rank: $rank"
done
echo ""
for target in "${APPLY_SVD_TO[@]}"; do
    echo "  - Apply to: $target layers"
done
echo ""

# ========================================
# 实验 1: 完整的两阶段训练（推荐）
# ========================================
echo "========================================="
echo "Experiment Set 1: Two-Stage Training"
echo "========================================="
echo ""

for seed in "${SEEDS[@]}"; do
    for rank in "${SVD_RANKS[@]}"; do
        for target in "${APPLY_SVD_TO[@]}"; do

            exp_name="two_stage_r${rank}_${target}_seed${seed}"
            job_name="svd_${exp_name}"
            output_path="${OUTPUT_DIR}/${exp_name}"

            echo "Submitting: $job_name"
            echo "  SVD Rank: $rank"
            echo "  Apply to: $target"
            echo "  Output: $output_path"

            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=h200ea
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

# 初始化conda
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"

# 环境变量
export WANDB_PROJECT=tinyvit_experiments
export CUDA_VISIBLE_DEVICES=0

# 切换到工作目录
cd ${WORK_DIR}

echo "=========================================="
echo "Job: ${job_name}"
echo "Started at: \$(date)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  SVD Rank: $rank"
echo "  Apply SVD to: $target"
echo "  Seed: $seed"
echo "  Mode: two_stage (Stage 1 + Stage 2)"
echo ""

python experiment_svd_continual_learning.py \
    --from_scratch \
    --experiment_mode two_stage \
    --svd_rank $rank \
    --apply_svd_to $target \
    --output_dir $output_path \
    --project_name svd_continual_learning \
    --run_name ${exp_name} \
    --stage1_per_device_train_batch_size $BATCH_SIZE \
    --stage1_gradient_accumulation_steps $GRAD_ACCUM \
    --stage1_learning_rate $STAGE1_LR \
    --stage1_max_steps $STAGE1_MAX_STEPS \
    --stage1_save_steps $STAGE1_SAVE_STEPS \
    --stage1_eval_steps $STAGE1_EVAL_STEPS \
    --stage1_logging_steps $STAGE1_LOGGING_STEPS \
    --stage2_per_device_train_batch_size $BATCH_SIZE \
    --stage2_gradient_accumulation_steps $GRAD_ACCUM \
    --stage2_learning_rate $STAGE2_LR \
    --stage2_max_steps $STAGE2_MAX_STEPS \
    --stage2_save_steps $STAGE2_SAVE_STEPS \
    --stage2_eval_steps $STAGE2_EVAL_STEPS \
    --stage2_logging_steps $STAGE2_LOGGING_STEPS \
    --max_length 1024 \
    --seed $seed \
    --device cuda

echo ""
echo "=========================================="
echo "Job: ${job_name}"
echo "Finished at: \$(date)"
echo "=========================================="
EOF

            echo "  ✓ Submitted"
            echo ""

        done
    done
done

# ========================================
# 实验 2: 只训练 Stage 1（用于后续分析）
# ========================================
echo "========================================="
echo "Experiment Set 2: Stage 1 Only (Optional)"
echo "========================================="
echo "Uncomment the following section if you want to train Stage 1 models separately"
echo ""

: <<'COMMENT'
for seed in "${SEEDS[@]}"; do
    for rank in "${SVD_RANKS[@]}"; do
        for target in "${APPLY_SVD_TO[@]}"; do

            exp_name="stage1_only_r${rank}_${target}_seed${seed}"
            job_name="svd_${exp_name}"
            output_path="${OUTPUT_DIR}/${exp_name}"

            echo "Submitting: $job_name"
            echo "  SVD Rank: $rank"
            echo "  Apply to: $target"
            echo "  Output: $output_path"

            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --time=24:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G


echo "=========================================="
echo "Job: ${job_name}"
echo "Started at: \$(date)"
echo "=========================================="

python experiment_svd_continual_learning.py \
    --experiment_mode stage1_only \
    --svd_rank $rank \
    --apply_svd_to $target \
    --output_dir $output_path \
    --project_name svd_continual_learning_stage1 \
    --run_name ${exp_name} \
    --stage1_per_device_train_batch_size $BATCH_SIZE \
    --stage1_gradient_accumulation_steps $GRAD_ACCUM \
    --stage1_learning_rate $STAGE1_LR \
    --stage1_max_steps $STAGE1_MAX_STEPS \
    --stage1_save_steps $STAGE1_SAVE_STEPS \
    --stage1_eval_steps $STAGE1_EVAL_STEPS \
    --stage1_logging_steps $STAGE1_LOGGING_STEPS \
    --max_length 1024 \
    --seed $seed \
    --device cuda

echo "=========================================="
echo "Job: ${job_name}"
echo "Finished at: \$(date)"
echo "=========================================="
EOF

            echo "  ✓ Submitted"
            echo ""

        done
    done
done
COMMENT

# ========================================
# 实验 3: 从已有 Stage 1 模型运行 Stage 2
# ========================================
echo "========================================="
echo "Experiment Set 3: Stage 2 from existing Stage 1 models (Optional)"
echo "========================================="
echo "Use this section to run Stage 2 on pre-trained Stage 1 models"
echo ""

: <<'COMMENT'
# 如果你已经有训练好的Stage 1模型，可以用这个脚本只运行Stage 2
declare -A STAGE1_MODELS
STAGE1_MODELS["r256_all"]="${OUTPUT_DIR}/two_stage_r256_all_seed42/stage1_two_stage_r256_all_seed42/final_model"
STAGE1_MODELS["r512_all"]="${OUTPUT_DIR}/two_stage_r512_all_seed42/stage1_two_stage_r512_all_seed42/final_model"

for model_key in "${!STAGE1_MODELS[@]}"; do
    stage1_path="${STAGE1_MODELS[$model_key]}"

    # 检查模型是否存在
    if [ ! -d "$stage1_path" ]; then
        echo "⚠️  Stage 1 model not found: $stage1_path"
        echo "   Skipping..."
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        exp_name="stage2_from_${model_key}_seed${seed}"
        job_name="svd_${exp_name}"
        output_path="${OUTPUT_DIR}/${exp_name}"

        echo "Submitting: $job_name"
        echo "  Stage 1 model: $stage1_path"
        echo "  Output: $output_path"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G


echo "=========================================="
echo "Job: ${job_name}"
echo "Started at: \$(date)"
echo "=========================================="

python experiment_svd_continual_learning.py \
    --experiment_mode stage2_only \
    --stage1_model_path $stage1_path \
    --svd_rank 256 \
    --apply_svd_to all \
    --output_dir $output_path \
    --project_name svd_continual_learning_stage2 \
    --run_name ${exp_name} \
    --stage2_per_device_train_batch_size $BATCH_SIZE \
    --stage2_gradient_accumulation_steps $GRAD_ACCUM \
    --stage2_learning_rate $STAGE2_LR \
    --stage2_max_steps $STAGE2_MAX_STEPS \
    --stage2_save_steps $STAGE2_SAVE_STEPS \
    --stage2_eval_steps $STAGE2_EVAL_STEPS \
    --stage2_logging_steps $STAGE2_LOGGING_STEPS \
    --max_length 1024 \
    --seed $seed \
    --device cuda

echo "=========================================="
echo "Job: ${job_name}"
echo "Finished at: \$(date)"
echo "=========================================="
EOF

        echo "  ✓ Submitted"
        echo ""

    done
done
COMMENT

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${LOGS_DIR}"
echo "Results will be saved in: ${OUTPUT_DIR}"
echo ""
echo "After completion, analyze results with:"
echo "  cd ${OUTPUT_DIR}"
echo "  ls -la */*/stage*_results.json"
echo ""
