#!/bin/bash

# ========================================
# 两阶段训练实验 - 仅Stage 2
# 使用已训练好的Stage 1模型
# ========================================

nvidia-smi

SEEDS=(4)
MODEL_SIZE="gpt2"
TRAIN_SAMPLES=36718
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_STEPS=40000

# 目录配置
BASE_DIR="/usr/project/newxtmp/hm235/random_transformer"
PRETRAINED_BASE="${BASE_DIR}/outputs/full_training_random_bp"
OUTPUT_DIR="${BASE_DIR}/outputs/full_training_random_bp_stage2_experiments"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# ========== 已有Stage 1模型路径配置 ==========
# 请根据你的实际文件结构修改这些路径
declare -A STAGE1_MODELS
STAGE1_MODELS[1]="${PRETRAINED_BASE}/full_train_both_resample_seed42/full_train_both_resample_seed42/final_model"
STAGE1_MODELS[2]="${PRETRAINED_BASE}/full_train_attn_resample_seed42/full_train_attn_resample_seed42/final_model"
STAGE1_MODELS[3]="${PRETRAINED_BASE}/full_train_mlp_resample_seed42/full_train_mlp_resample_seed42/final_model"


# 策略名称映射
declare -A STRATEGY_NAMES
STRATEGY_NAMES[1]="freeze_attn_mlp"
STRATEGY_NAMES[2]="freeze_attn_only"
STRATEGY_NAMES[3]="freeze_mlp_only"
STRATEGY_NAMES[297]="train_mlp_only"
STRATEGY_NAMES[298]="train_attn_only"
STRATEGY_NAMES[299]="train_attn_mlp_only"

# 实验配置
# 格式: exp_name:stage1_strategy:stage2_strategy:description
declare -A EXPERIMENTS

# ========== 从策略2的模型开始 ==========
# EXPERIMENTS[10]="s1_freeze_attn_s2_train_attn_only:2:298:S1(freeze Attn) → S2(train Attn only)"
# EXPERIMENTS[11]="s1_freeze_attn_s2_freeze_mlp:2:3:S1(freeze Attn) → S2(freeze MLP, keep training Attn+Emb)"
# EXPERIMENTS[12]="s1_freeze_attn_s2_train_mlp_only:2:297:S1(freeze Attn) → S2(train MLP only)"
EXPERIMENTS[13]="s1_freeze_attn_s2_fully_training:2:0:S1(freeze Attn) → S2(train everything)"

# ========== 从策略3的模型开始 ==========
# EXPERIMENTS[20]="s1_freeze_mlp_s2_train_mlp_only:3:297:S1(freeze MLP) → S2(train MLP only)"
# EXPERIMENTS[21]="s1_freeze_mlp_s2_freeze_attn:3:2:S1(freeze MLP) → S2(freeze Attn, keep training MLP+Emb)"
# EXPERIMENTS[22]="s1_freeze_mlp_s2_train_attn_only:3:298:S1(freeze MLP) → S2(train Attn only)"
EXPERIMENTS[23]="s1_freeze_mlp_s2_fully_training:3:0:S1(freeze MLP) → S2(train everything)"

# ========== 从策略1的模型开始 ==========
# EXPERIMENTS[30]="s1_freeze_attn_mlp_s2_train_attn_mlp:1:299:S1(freeze Attn+MLP) → S2(train Attn+MLP)"
EXPERIMENTS[31]="s1_freeze_attn_mlp_s2_fully_training:1:0:S1(freeze Attn+MLP) → S2(train everything)"

echo "========================================"
echo "Stage 2 Experiments - Readable Folder Names"
echo "========================================"
echo ""
echo "Strategy System:"
echo "  Original: 1=freeze_attn_mlp, 2=freeze_attn_only, 3=freeze_mlp_only"
echo "  Symmetric: 297=train_mlp_only, 298=train_attn_only, 299=train_attn_mlp_only"
echo "  Sum check: 1+299=300, 2+298=300, 3+297=300"
echo ""

# 检查已有模型
echo "Verifying existing Stage 1 models:"
all_exist=true
for strat in 1 2 3; do
    if [ -d "${STAGE1_MODELS[$strat]}" ]; then
        echo "  ✓ Strategy ${strat} (${STRATEGY_NAMES[$strat]}): Found"
    else
        echo "  ✗ Strategy ${strat} (${STRATEGY_NAMES[$strat]}): NOT FOUND"
        echo "     Expected at: ${STAGE1_MODELS[$strat]}"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = false ]; then
    echo "⚠️  ERROR: Some Stage 1 models are missing!"
    echo "Please update STAGE1_MODELS paths in the script."
    echo "Aborting..."
    exit 1
fi

echo "All Stage 1 models found. Submitting Stage 2 experiments..."
echo ""

for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name stage1_strategy stage2_strategy description <<< "${EXPERIMENTS[$exp_key]}"
    
    s1_name="${STRATEGY_NAMES[$stage1_strategy]}"
    s2_name="${STRATEGY_NAMES[$stage2_strategy]}"
    
    echo "Experiment $exp_key: $description"
    echo "  Folder name: ${exp_name}"
    echo "  Stage 1: ${s1_name} (strategy ${stage1_strategy})"
    echo "  Stage 2: ${s2_name} (strategy ${stage2_strategy})"
    
    for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        stage1_model_path="${STAGE1_MODELS[$stage1_strategy]}"
        output_path="${OUTPUT_DIR}/${exp_name}_seed${seed}"
        
        echo "  - Submitting: $job_name"
        echo "    Output: ${output_path}"
        
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${job_name}_%j.err
#SBATCH --gres=gpu:a6000:1 -p compsci-gpu 
#SBATCH --mem=100G
#SBATCH --time=48:00:00

export WANDB_PROJECT=stage2_experiments
export WANDB_TAGS="stage2,s1_${s1_name},s2_${s2_name},exp_${exp_key}"

echo "========================================"
echo "Stage 2 Training"
echo "Experiment: ${exp_name}"
echo "Stage 1 strategy: ${s1_name} (${stage1_strategy})"
echo "Stage 2 strategy: ${s2_name} (${stage2_strategy})"
echo "Loading from: ${stage1_model_path}"
echo "Output to: ${output_path}"
echo "========================================"

python experiment_llm.py \\
    --load_pretrained_path ${stage1_model_path} \\
    --model_size ${MODEL_SIZE} \\
    --weight_frozen ${stage2_strategy} \\
    --stage 2 \\
    --seed ${seed} \\
    --train_samples ${TRAIN_SAMPLES} \\
    --per_device_train_batch_size ${BATCH_SIZE} \\
    --gradient_accumulation_steps ${GRAD_ACCUM} \\
    --per_device_eval_batch_size 4 \\
    --learning_rate 5e-4 \\
    --weight_decay 0.01 \\
    --max_steps ${MAX_STEPS} \\
    --eval_nq_samples 1000 \\
    --eval_lambada_samples 1000 \\
    --eval_wikitext_samples 1000 \\
    --max_length 1024 \\
    --run_name ${job_name} \\
    --output_dir ${output_path}

echo ""
echo "========================================"
echo "Stage 2 completed: ${job_name}"
echo "Finished at: \$(date)"
echo "========================================"
EOF
        
        sleep 0.3
    done
    echo ""
done

echo "========================================"
echo "All Stage 2 experiments submitted!"
echo "========================================"
echo ""
echo "Output directory structure:"
echo "${OUTPUT_DIR}/"
echo "├── s1_freeze_attn_s2_train_attn_only_seed42/"
echo "├── s1_freeze_attn_s2_freeze_mlp_seed42/"
echo "├── s1_freeze_attn_s2_train_mlp_only_seed42/"
echo "├── s1_freeze_mlp_s2_train_mlp_only_seed42/"
echo "├── s1_freeze_mlp_s2_freeze_attn_seed42/"
echo "├── s1_freeze_mlp_s2_train_attn_only_seed42/"
echo "└── s1_freeze_attn_mlp_s2_train_attn_mlp_seed42/"
echo ""
echo "Experiment summary:"
echo "  Group 1 (S1: freeze_attn_only): 3 experiments"
echo "    [10] → train_attn_only    (symmetric reversal)"
echo "    [11] → freeze_mlp_only     (continue with Emb trainable)"
echo "    [12] → train_mlp_only      (switch component)"
echo ""
echo "  Group 2 (S1: freeze_mlp_only): 3 experiments"
echo "    [20] → train_mlp_only      (symmetric reversal)"
echo "    [21] → freeze_attn_only    (continue with Emb trainable)"
echo "    [22] → train_attn_only     (switch component)"
echo ""
echo "  Group 3 (S1: freeze_attn_mlp): 1 experiment"
echo "    [30] → train_attn_mlp_only (symmetric reversal)"
echo ""
echo "Total jobs: 7"
echo ""
echo "Key comparisons:"
echo "  • Symmetric pairs: [10] vs [20] (component reversal)"
echo "  • Embedding timing: [11] vs [10], [21] vs [20]"
echo "  • Training order: [12] vs [22]"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs: ${LOGS_DIR}"
echo "WandB: stage2_experiments"