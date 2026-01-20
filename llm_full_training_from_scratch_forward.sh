#!/bin/bash

# ============================================================================
# Random Forward Propagation 实验
# 策略A: Additive Noise (W_forward = W_real + noise)
# 策略B: Full Random (W_forward = W_random, Feedback Alignment style)
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
OUTPUT_DIR="${BASE_DIR}/outputs/random_forward"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# 实验配置
# 格式: exp_name:freeze_strategy:random_forward_strategy:apply_to:noise_scale:description
declare -A EXPERIMENTS

# ============================================================================
# 策略A: Additive Noise (推荐，数学上正确)
# ============================================================================

# A1: Freeze Attn (训练MLP+Emb), random forward on frozen Attn
EXPERIMENTS[A1]="freeze_attn_rfa_attn:2:additive_noise:attn:0.02:Freeze Attn, additive noise on frozen Attn"

# A2: Freeze MLP (训练Attn+Emb), random forward on frozen MLP
EXPERIMENTS[A2]="freeze_mlp_rfa_mlp:3:additive_noise:mlp:0.02:Freeze MLP, additive noise on frozen MLP"

# A4: Full finetune with additive noise on all layers (测试正则化效果)
EXPERIMENTS[A3]="full_finetune_rfa_all:0:additive_noise:all:0.02:Full finetune, additive noise on all"


# ============================================================================
# 策略B: Full Random (Feedback Alignment style)
# ============================================================================

# B1: Freeze Attn (训练MLP+Emb), full random forward on frozen Attn
EXPERIMENTS[B1]="freeze_attn_rfb_attn:2:full_random:attn::Freeze Attn, full random on frozen Attn"

# B2: Freeze MLP (训练Attn+Emb), full random forward on frozen MLP
EXPERIMENTS[B2]="freeze_mlp_rfb_mlp:3:full_random:mlp::Freeze MLP, full random on frozen MLP"

# B4: Full finetune with full random on all layers
EXPERIMENTS[B3]="full_finetune_rfb_all:0:full_random:all::Full finetune, full random on all"


echo "========================================================================"
echo "Training with Random Forward Propagation Strategies"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Model: ${MODEL_SIZE}"
echo ""
echo "Strategy A: Additive Noise (W_forward = W_real + noise)"
echo "Strategy B: Full Random (W_forward = W_random, Feedback Alignment)"
echo ""
echo "Total experiments: $((${#EXPERIMENTS[@]} * 2)) (${#EXPERIMENTS[@]} configs × 2 modes)"
echo ""

# 遍历所有实验
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r base_exp_name freeze_strategy rf_strategy rf_apply noise_scale base_description <<< "${EXPERIMENTS[$exp_key]}"
    
    # 为每个实验生成fixed和resample两个版本
    for mode in "fixed" "resample"; do
        exp_name="${base_exp_name}_${mode}"
        
        if [ "$mode" = "resample" ]; then
            resample_flag="--forward_resample_every_batch"
            mode_desc="resample every batch"
        else
            resample_flag=""
            mode_desc="fixed random weights/noise"
        fi
        
        echo "======================================================================"
        echo "Experiment ${exp_key}-${mode}: ${base_description} [${mode}]"
        echo "======================================================================"
        echo "  Freeze strategy: ${freeze_strategy}"
        echo "  Random Forward: ${rf_strategy} on ${rf_apply}"
        if [ -n "$noise_scale" ]; then
            echo "  Noise scale: ${noise_scale}"
        fi
        echo "  Mode: ${mode_desc}"
        echo ""
        
        for seed in "${SEEDS[@]}"; do
            job_name="${exp_name}_seed${seed}"
            output_dir="${OUTPUT_DIR}/${exp_name}_seed${seed}"
            
            echo "  Submitting: $job_name"
            
            # 构建 noise_scale 参数
            if [ -n "$noise_scale" ]; then
                noise_scale_arg="--forward_noise_scale ${noise_scale}"
            else
                noise_scale_arg=""
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
#SBATCH --time=15:00:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

cd ${WORK_DIR}

source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"
 
python experiment_llm.py \
    --model_size ${MODEL_SIZE} \
    --seed ${seed} \
    --weight_frozen ${freeze_strategy} \
    --from_scratch \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --weight_decay 0.01 \
    --eval_nq_samples 1000 \
    --eval_lambada_samples 1000 \
    --eval_wikitext_samples 1000 \
    --max_length 1024 \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --output_dir ${output_dir} \
    --project_name "random_forward_exp" \
    --run_name "${job_name}" \
    --random_forward_strategy ${rf_strategy} \
    --apply_random_forward_to_layers ${rf_apply} \
    --forward_projection_type random \
    ${noise_scale_arg} \
    ${resample_flag}

echo "Job completed: ${job_name}"
EOF

        done
        echo ""
    done
done

echo "========================================================================"
echo "All jobs submitted!"
echo "========================================================================"
echo ""
echo "Summary of experiments:"
echo "  Strategy A (Additive Noise): ${#EXPERIMENTS[@]} configs with A prefix"
echo "  Strategy B (Full Random): ${#EXPERIMENTS[@]} configs with B prefix"
echo "  Each config × 2 modes (fixed/resample) × ${#SEEDS[@]} seeds"
echo "  Total jobs: $((${#EXPERIMENTS[@]} * 2 * ${#SEEDS[@]}))"
echo "========================================================================"
