#!/bin/bash

# ============================================================================
# Langevin Dynamics Noise Injection 实验
# 测试对不同layer的梯度添加噪声的效果
# 对比: Embedding / Attention / MLP / All layers
# ============================================================================

nvidia-smi

# 基础配置
SEEDS=(42)
MODEL_SIZE="gpt2"
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_STEPS=80000
LEARNING_RATE=5e-4

# 目录配置
BASE_DIR="/work/hm235/random_transformer"
OUTPUT_DIR="${BASE_DIR}/outputs/langevin_noise_layers"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# 实验配置
# 格式: exp_name:freeze_strategy:noise_apply_to:noise_scale:temperature:use_precond:description
declare -A EXPERIMENTS


# ============================================================================
# 组B: 全参数训练 + 对不同层加噪声（测试正则化效果）
# ============================================================================

# B1: Full training, 只对Embedding加噪声
# EXPERIMENTS[B1]="full_noise_emb:0:embedding:0.02:1.0:false:Full training, noise on Emb only"

# # B2: Full training, 只对Attention加噪声
EXPERIMENTS[B2]="full_noise_attn_noise_0.02:0:attn:0.02:1.0:false:Full training, noise on Attn only"

# # B3: Full training, 只对MLP加噪声
# EXPERIMENTS[B3]="full_noise_mlp:0:mlp:0.02:1.0:false:Full training, noise on MLP only"

# B4: Full training, 对所有层加噪声
# EXPERIMENTS[B4]="full_noise_all:0:all:0.02:1.0:false:Full training, noise on all layers"

# ============================================================================
# 组C: 不同噪声强度测试
# ============================================================================

# C1: 小噪声 (0.001)
# EXPERIMENTS[C1]="full_noise_attn:0:attn:0.001:1.0:false:Full training, small noise (0.001)"

# C2: 中等噪声 (0.02) - 与A1重复，用于对比
# EXPERIMENTS[C2]="freeze_both_noise_emb_medium:1:embedding:0.02:1.0:false:Freeze Both, medium noise (0.02)"

# C3: 大噪声 (0.05)
# EXPERIMENTS[C3]="full_noise_attn_noise_0.1:0:attn:0.1:1.0:false:Full training, large noise (0.1)"

# ============================================================================
# 组D: Preconditioner测试（自适应噪声）
# ============================================================================

# D2: Full training + Preconditioner
# EXPERIMENTS[D2]="full_noise_all_precond:0:all:0.02:1.0:true:Full training, all layers with precond"


echo "========================================================================"
echo "Langevin Dynamics Noise Injection Experiments"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Model: ${MODEL_SIZE}"
echo "  Seeds: ${SEEDS[@]}"
echo ""
echo "Experiment Groups:"
echo "  Group A: 冻结策略 + 噪声 (对应Random BP场景)"
echo "  Group B: 全参数训练 + 不同层噪声 (正则化效果)"
echo "  Group C: 噪声强度对比 (0.001 vs 0.02 vs 0.1)"
echo "  Group D: Preconditioner测试 (自适应噪声)"
echo "  Group E: Temperature测试 (探索vs利用)"
echo ""
echo "Total experiments: ${#EXPERIMENTS[@]} configs × ${#SEEDS[@]} seeds = $((${#EXPERIMENTS[@]} * ${#SEEDS[@]})) jobs"
echo ""

# 遍历所有实验
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name freeze_strategy noise_apply noise_scale temperature use_precond description <<< "${EXPERIMENTS[$exp_key]}"

    echo "======================================================================"
    echo "Experiment ${exp_key}: ${description}"
    echo "======================================================================"
    echo "  Freeze strategy: ${freeze_strategy}"
    echo "  Noise apply to: ${noise_apply}"
    echo "  Noise scale: ${noise_scale}"
    echo "  Temperature: ${temperature}"
    echo "  Preconditioner: ${use_precond}"
    echo ""

    for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        output_dir="${OUTPUT_DIR}/${exp_name}_seed${seed}"

        echo "  Submitting: ${job_name}"

        # 构建preconditioner参数
        if [ "$use_precond" = "true" ]; then
            precond_flag="--langevin_precond"
        else
            precond_flag=""
        fi

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=h200ea
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

cd ${WORK_DIR}

source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"

python experiment_llm_langevin.py \
    --model_size ${MODEL_SIZE} \
    --seed ${seed} \
    --weight_frozen ${freeze_strategy} \
    --from_scratch \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.01 \
    --output_dir ${output_dir} \
    --project_name "langevin_noise_layers" \
    --run_name "${job_name}" \
    --max_length 1024 \
    --eval_nq_samples 1000 \
    --eval_lambada_samples 1000 \
    --eval_wikitext_samples 1000 \
    --use_langevin_baseline \
    --langevin_noise_scale ${noise_scale} \
    --langevin_apply_to ${noise_apply} \
    ${precond_flag}

echo "Job completed: ${job_name}"
EOF

    done
    echo ""
done

echo "========================================================================"
echo "All jobs submitted!"
echo "========================================================================"
echo ""
echo "Experiment Summary:"
echo "  Group A (冻结+噪声): 3 configs"
echo "  Group B (全参数+层级噪声): 4 configs"
echo "  Group C (噪声强度): 2 configs"
echo "  Group D (Preconditioner): 2 configs"
echo "  Group E (Temperature): 2 configs"
echo "  Total: ${#EXPERIMENTS[@]} configs × ${#SEEDS[@]} seeds = $((${#EXPERIMENTS[@]} * ${#SEEDS[@]})) jobs"
echo ""
echo "Monitor jobs: squeue -u \$USER"
echo "View logs: tail -f ${LOGS_DIR}/<job_name>.out"
echo "========================================================================"
