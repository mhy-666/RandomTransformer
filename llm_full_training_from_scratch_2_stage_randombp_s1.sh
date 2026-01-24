#!/bin/bash

# ============================================================================
# Simplified: 冻结策略 + Random Backpropagation 实验
# 只对冻结的layer进行random BP
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
OUTPUT_DIR="${BASE_DIR}/outputs/freeze_random_bp_2"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# 实验配置
# 格式: exp_name:freeze_strategy:random_bp_apply:description
declare -A EXPERIMENTS

# Freeze Attn (训练MLP+Emb), random BP on frozen Attn
EXPERIMENTS[1]="freeze_attn_rbp_attn:2:attn:Freeze Attn, random BP on frozen Attn"

# # Freeze MLP (训练Attn+Emb), random BP on frozen MLP
# EXPERIMENTS[2]="freeze_mlp_rbp_mlp:3:mlp:Freeze MLP, random BP on frozen MLP"

# # Freeze Both (只训练Emb), random BP on frozen Attn+MLP
# EXPERIMENTS[3]="freeze_both_rbp_both:1:attn_mlp:Freeze Both, random BP on frozen Attn+MLP"

# # Fully training, random BP on Attn
# EXPERIMENTS[4]="full_train_attn:0:attn:Full train - random BP on Attn"

# # Fully training, random BP on MLP
# EXPERIMENTS[5]="full_train_attn:0:mlp:Full train - random BP on MLP"

# # Fully training, random BP on Attn+MLP
# EXPERIMENTS[5]="full_train_attn:0:attn_mlp:Full train - random BP on MLP"



echo "========================================================================"
echo "Training with Freeze Strategies + Random Backpropagation"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Model: ${MODEL_SIZE}"
echo ""
echo "Total experiments: $((${#EXPERIMENTS[@]} * 2)) (3 base × 2 modes)"
echo ""

# 遍历所有实验
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r base_exp_name freeze_strategy random_bp_apply base_description <<< "${EXPERIMENTS[$exp_key]}"
    
    # 为每个实验生成fixed和resample两个版本
    for mode in "fixed" "resample"; do
        exp_name="${base_exp_name}_${mode}"
        
        if [ "$mode" = "resample" ]; then
            resample_flag="--resample_every_batch"
            mode_desc="resample every batch"
        else
            resample_flag=""
            mode_desc="fixed random params"
        fi
        
        echo "======================================================================"
        echo "Experiment ${exp_key}-${mode}: ${base_description} [${mode}]"
        echo "======================================================================"
        echo "  Freeze strategy: ${freeze_strategy}, Random BP on: ${random_bp_apply}"
        echo "  Mode: ${mode_desc}"
        echo ""
        
        for seed in "${SEEDS[@]}"; do
            job_name="${exp_name}_seed${seed}"
            output_dir="${OUTPUT_DIR}/${exp_name}_seed${seed}"
            
            echo "  Submitting: $job_name"
            
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=h200ea
#SBATCH --output=${LOGS_DIR}/${job_name}.out
#SBATCH --error=${LOGS_DIR}/${job_name}.err
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
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
    --project_name "freeze_random_bp" \
    --run_name "${job_name}" \
    --random_backprop_strategy full_random \
    --apply_random_backprop_to_layers ${random_bp_apply} \
    --projection_type random \
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
