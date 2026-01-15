#!/bin/bash

# ============================================================================
# Random Backprop - Full Parameter Training Experiments
# 全参数训练场景下的Random Backprop实验
# ============================================================================


# 定义基础配置
SEEDS=(42)
MODEL_SIZE="gpt2"
TRAIN_SAMPLES=36718
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_STEPS=80000
LEARNING_RATE=5e-4
OUTPUT_DIR="/work/hm235/random_transformer/outputs/full_training_random_bp"
LOGS_DIR="${OUTPUT_DIR}/logs"

mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# 实验配置列表
# 格式: exp_name:weight_frozen:apply_layers:random_bp_strategy:proj_type:proj_rank:resample:disable_ratio:description
declare -A EXPERIMENTS

# ============================================================================
# Group 1: Baseline实验（对照组）
# ============================================================================
EXPERIMENTS[0]="baseline_full_train:0:all:none:random:0:false:1.0:Baseline - Full training, standard backprop"

# ============================================================================
# Group 2: 全参数训练 + Random BP (MLP only)
# ============================================================================
# 固定随机参数
EXPERIMENTS[10]="full_train_mlp_fixed:0:mlp:full_random:random:0:false:1.0:Full train - MLP with fixed random backprop"

# 每batch重采样
EXPERIMENTS[11]="full_train_mlp_resample:0:mlp:full_random:random:0:true:1.0:Full train - MLP with resample random backprop"

# 低秩投影 rank=256
EXPERIMENTS[12]="full_train_mlp_lowrank256:0:mlp:low_rank_projection:random:256:false:1.0:Full train - MLP with low-rank projection r=256"

# 低秩投影 rank=512
EXPERIMENTS[13]="full_train_mlp_lowrank512:0:mlp:low_rank_projection:random:512:false:1.0:Full train - MLP with low-rank projection r=512"

# 旋转矩阵 rank=256
EXPERIMENTS[14]="full_train_mlp_rotation256:0:mlp:low_rank_projection:rotation:256:false:1.0:Full train - MLP with rotation matrix r=256"

# ============================================================================
# Group 3: 全参数训练 + Random BP (Attention only)
# ============================================================================
EXPERIMENTS[20]="full_train_attn_fixed:0:attn:full_random:random:0:false:1.0:Full train - Attn with fixed random backprop"

EXPERIMENTS[21]="full_train_attn_resample:0:attn:full_random:random:0:true:1.0:Full train - Attn with resample random backprop"

EXPERIMENTS[22]="full_train_attn_lowrank256:0:attn:low_rank_projection:random:256:false:1.0:Full train - Attn with low-rank projection r=256"

# ============================================================================
# Group 4: 全参数训练 + Random BP (Both Attn+MLP)
# ============================================================================
EXPERIMENTS[30]="full_train_both_fixed:0:attn_mlp:full_random:random:0:false:1.0:Full train - Both with fixed random backprop"

EXPERIMENTS[31]="full_train_both_resample:0:attn_mlp:full_random:random:0:true:1.0:Full train - Both with resample random backprop"

# ============================================================================
# Group 5: 最后10%关闭Random BP（重点实验）
# ============================================================================
EXPERIMENTS[40]="full_train_mlp_disable90:0:mlp:full_random:random:0:false:0.9:Full train - MLP, disable random BP at 90%"

EXPERIMENTS[41]="full_train_mlp_lowrank_disable90:0:mlp:low_rank_projection:random:256:false:0.9:Full train - MLP low-rank, disable at 90%"

EXPERIMENTS[42]="full_train_attn_disable90:0:attn:full_random:random:0:false:0.9:Full train - Attn, disable random BP at 90%"

EXPERIMENTS[43]="full_train_both_disable90:0:attn_mlp:full_random:random:0:false:0.9:Full train - Both, disable random BP at 90%"

# ============================================================================
# Group 6: 不同关闭时间点的消融实验
# ============================================================================
EXPERIMENTS[50]="full_train_mlp_disable80:0:mlp:full_random:random:0:false:0.8:Full train - MLP, disable at 80%"

EXPERIMENTS[51]="full_train_mlp_disable95:0:mlp:full_random:random:0:false:0.95:Full train - MLP, disable at 95%"

echo "========================================================================"
echo "Random Backpropagation - Full Parameter Training Experiments"
echo "========================================================================"
echo ""
echo "Experiment Groups:"
echo "  [0]     Baseline - Standard full training"
echo "  [10-14] Full training + Random BP (MLP only)"
echo "  [20-22] Full training + Random BP (Attention only)"
echo "  [30-31] Full training + Random BP (Both Attn+MLP)"
echo "  [40-43] Disable Random BP at 90% (Key experiments)"
echo "  [50-51] Ablation - Different disable ratios"
echo ""
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo ""

# 遍历所有实验配置
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name weight_frozen apply_layers random_bp_strategy proj_type proj_rank resample disable_ratio description <<< "${EXPERIMENTS[$exp_key]}"

    echo "========================================================================"
    echo "Experiment $exp_key: $description"
    echo "========================================================================"
    echo "  Weight frozen: $weight_frozen (0=full training)"
    echo "  Apply random BP to: $apply_layers"
    echo "  Strategy: $random_bp_strategy"

    if [ "$random_bp_strategy" != "none" ]; then
        echo "  Projection type: $proj_type"
        if [ "$proj_rank" -ne 0 ]; then
            echo "  Projection rank: $proj_rank"
        else
            echo "  Projection rank: full"
        fi
        echo "  Resample: $resample"
        echo "  Disable at ratio: $disable_ratio (1.0=never disable)"
    fi
    echo ""

    # 遍历所有 seeds
    for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        echo "  [Submitting] $job_name"

        # 构建随机反向传播参数
        random_bp_params="--random_backprop_strategy ${random_bp_strategy}"

        if [ "$random_bp_strategy" != "none" ]; then
            random_bp_params="${random_bp_params} --apply_random_backprop_to_layers ${apply_layers}"
            random_bp_params="${random_bp_params} --projection_type ${proj_type}"

            if [ "$proj_rank" -ne 0 ]; then
                random_bp_params="${random_bp_params} --projection_rank ${proj_rank}"
            fi

            if [ "$resample" = "true" ]; then
                random_bp_params="${random_bp_params} --resample_every_batch"
            fi

            # 添加disable ratio参数
            if (( $(echo "$disable_ratio < 1.0" | bc -l) )); then
                random_bp_params="${random_bp_params} --disable_random_bp_at_ratio ${disable_ratio}"
            fi
        fi

        # 提交SLURM任务
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=h200ea
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${job_name}_%j.err
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

# 运行实验
python experiment_llm.py \
    --weight_frozen ${weight_frozen} \
    --from_scratch \
    --model_size ${MODEL_SIZE} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --per_device_eval_batch_size 4 \
    --weight_decay 0.01 \
    --eval_nq_samples 1000 \
    --eval_lambada_samples 1000 \
    --eval_wikitext_samples 1000 \
    --max_length 1024 \
    --seed ${seed} \
    --run_name ${job_name} \
    --output_dir ${OUTPUT_DIR}/${job_name} \
    --project_name "random_bp_full_training" \
    ${random_bp_params}

echo "Job ${job_name} completed"
EOF

    done
    echo ""
done

echo "========================================================================"
echo "All jobs submitted!"
echo "========================================================================"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${LOGS_DIR}"
echo "Results in: ${OUTPUT_DIR}"
