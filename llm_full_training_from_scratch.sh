#!/bin/bash

# 创建必要的目录
# nvidia-smi

# 定义实验配置
SEEDS=(4)
MODEL_SIZE="gpt2"
TRAIN_SAMPLES=36718 
BATCH_SIZE=4
GRAD_ACCUM=4
OUTPUT_DIR="/work/hm235/random_transformer/outputs/from_scratch_29"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}


# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"
# 实验配置列表
# 格式: exp_name:weight_frozen:qkv_init:alpha:beta:freeze_qkv_comp:qkv_init_comp:description
declare -A EXPERIMENTS

# # ========== 基线实验 (原始配置) ==========
EXPERIMENTS[0]="baseline_0_zeroshot:-1:all:0.8:0.2:none:none:Zero-shot (no training)"
EXPERIMENTS[1]="baseline_1_full_finetune:0:all:0.8:0.2:none:none:Full parameter fine-tuning"
EXPERIMENTS[2]="baseline_2_freeze_attn:2:all:0.8:0.2:none:none:Freeze Attention only"
EXPERIMENTS[3]="baseline_3_freeze_mlp:3:all:0.8:0.2:none:none:Freeze MLP only"
EXPERIMENTS[4]="exp_1_freeze_attn_mlp:1:all:0.8:0.2:none:none:Freeze Attention+MLP"
EXPERIMENTS[5]="exp_2_freeze_attn_mlp_pos:100:all:0.8:0.2:none:none:Freeze Attention+MLP+Pos"

# ========== QKV单位矩阵初始化实验 ==========
# EXPERIMENTS[10]="qkv_all_full_finetune:0:all:1.0:0.0:none:none:QKV Identity All Layers + Full Finetune"
# EXPERIMENTS[11]="qkv_all_freeze_attn_mlp:1:all:1.0:0.0:none:none:QKV Identity All Layers + Freeze Attn+MLP"
# EXPERIMENTS[12]="qkv_all_freeze_attn:2:all:1.0:0.0:none:none:QKV Identity All Layers + Freeze Attn"
# EXPERIMENTS[13]="qkv_all_freeze_mlp:3:all:1.0:0.0:none:none:QKV Identity All Layers + Freeze MLP"

# ========== QKV细粒度控制实验 ==========
# Freeze Q and K, train V
# EXPERIMENTS[10]="qkv_freeze_qk_train_v:2:none:0.8:0.2:qk:none:Freeze Q+K, Train V"
# EXPERIMENTS[11]="qkv_freeze_qk_identity:2:all:1.0:0.0:qk:qk\:identity,v\:random:Freeze Q+K (Identity), Train V (Random)"
# EXPERIMENTS[12]="qkv_freeze_qk_random:2:all:1.0:0.0:qk:qk\:random,v\:random:Freeze Q+K (Random), Train V (Random)"
# EXPERIMENTS[12]="qkv_freeze_qk_mixed:2:mixed:0.8:0.2:qk:qk\:mixed,v\:random:Freeze Q+K (Mixed), Train V"

# Freeze K and V, train Q
# EXPERIMENTS[20]="qkv_freeze_kv_train_q:2:none:0.8:0.2:kv:none:Freeze K+V, Train Q"
# EXPERIMENTS[21]="qkv_freeze_kv_identity:2:all:1.0:0.0:kv:kv\:identity,q\:random:Freeze K+V (Identity), Train Q (Random)"
# EXPERIMENTS[22]="qkv_freeze_kv_random:2:all:1.0:0.0:kv:kv\:random,q\:random:Freeze K+V (Random), Train Q (Random)"

# Freeze Q and V, train K
# EXPERIMENTS[30]="qkv_freeze_qv_train_k:2:none:0.8:0.2:qv:none:Freeze Q+V, Train K"
# EXPERIMENTS[31]="qkv_freeze_qv_identity:2:all:1.0:0.0:qv:qv\:identity,k\:random:Freeze Q+V (Identity), Train K (Random)"
# EXPERIMENTS[32]="qkv_freeze_qv_random:2:all:1.0:0.0:qv:qv\:random,k\:random:Freeze Q+V (Identity), Train K (Random)"

# Only freeze Q
# EXPERIMENTS[40]="qkv_freeze_q_only:2:none:0.8:0.2:q:none:Freeze Q only, Train K+V"
# EXPERIMENTS[41]="qkv_freeze_q_identity:2:all:1.0:0.0:q:q\:identity,kv\:random:Freeze Q (Identity), Train K+V (Random)"
# EXPERIMENTS[42]="qkv_freeze_q_random:2:all:1.0:0.0:q:q\:random,kv\:random:Freeze Q (Random), Train K+V (Random)"
# Only freeze K
# EXPERIMENTS[50]="qkv_freeze_k_only:2:none:0.8:0.2:k:none:Freeze K only, Train Q+V"
# EXPERIMENTS[51]="qkv_freeze_k_identity:2:all:1.0:0.0:k:k\:identity,qv\:random:Freeze K (Identity), Train Q+V (Random)"
# EXPERIMENTS[52]="qkv_freeze_k_random:2:all:1.0:0.0:k:k\:random,qv\:random:Freeze K (Random), Train Q+V (Random)"
# Only freeze V
# EXPERIMENTS[60]="qkv_freeze_v_only:2:none:0.8:0.2:v:none:Freeze V only, Train Q+K"
# EXPERIMENTS[61]="qkv_freeze_v_identity:2:all:1.0:0.0:v:v\:identity,qk\:random:Freeze V (Identity), Train Q+K (Random)"
# EXPERIMENTS[62]="qkv_freeze_v_random:2:all:1.0:0.0:v:v\:random,qk\:random:Freeze V (Random), Train Q+K (Random)"
# # ========== 差异化初始化实验 ==========
# # Q用identity，K和V用mixed
# EXPERIMENTS[70]="qkv_q_id_kv_mixed:2:mixed:0.8:0.2:all:q\:identity,kv\:mixed:Q Identity + K,V Mixed"

# # Q和K用identity，V用random
# EXPERIMENTS[71]="qkv_qk_id_v_random:2:all:1.0:0.0:all:qk\:identity,v\:random:Q,K Identity + V Random"

# # 全random（对照组）
# EXPERIMENTS[72]="qkv_all_random:2:none:0.8:0.2:all:all\:random:All Random (control)"

# # ========== 浅层实验 ==========
# EXPERIMENTS[80]="qkv_shallow_freeze_qk:2:shallow:1.0:0.0:qk:qk\:identity,v\:random:Shallow Freeze Q+K"

echo "========================================"
echo "Submitting QKV Fine-grained Control Experiments"
echo "========================================"
echo ""

# 遍历所有实验配置
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name weight_frozen qkv_init alpha beta freeze_qkv_comp qkv_init_comp description <<< "${EXPERIMENTS[$exp_key]}"
    
    echo "Experiment $exp_key: $description"
    echo "  Freeze strategy: weight_frozen=$weight_frozen"
    echo "  QKV components to freeze: $freeze_qkv_comp"
    echo "  QKV init config: $qkv_init_comp"
    echo "  QKV init strategy: $qkv_init (alpha=$alpha, beta=$beta)"
    
    # 遍历所有 seeds
    for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        
        echo "  - Submitting job: $job_name"
        
        # 对于 zero-shot (weight_frozen=-1)，添加 --skip_training 标志
        if [ "$weight_frozen" -eq -1 ]; then
            skip_flag="--skip_training"
        else
            skip_flag=""
        fi
        
        # 处理 freeze_qkv_comp 参数
        if [ "$freeze_qkv_comp" = "none" ]; then
            freeze_qkv_param=""
        else
            freeze_qkv_param="--freeze_qkv_components ${freeze_qkv_comp}"
        fi
        
        # 处理 qkv_init_comp 参数（需要去掉转义符）
        if [ "$qkv_init_comp" = "none" ]; then
            qkv_init_param=""
        else
            # 去掉转义符 \: 变成 :
            qkv_init_clean=$(echo "$qkv_init_comp" | sed 's/\\:/:/g')
            qkv_init_param="--qkv_init_components \"${qkv_init_clean}\""
        fi


        # 自动判定权重共享参数
        share_flags=""

        if [ "$weight_frozen" -eq 2 ]; then
            share_flags="--share_attention_weights"
        elif [ "$weight_frozen" -eq 3 ]; then
            share_flags="--share_mlp_weights"
        elif [ "$weight_frozen" -eq 100 ]; then
            share_flags="--share_attention_weights --share_mlp_weights"
        else
            share_flags=""
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
#SBATCH --time=24:00:00
#SBATCH --partition=h200ea
#SBATCH --qos=normal

# 初始化conda
source ${CONDA_PATH}/bin/activate
conda activate ${CONDA_ENV}

export HF_DATASETS_CACHE="/work/hm235/hf_cache/datasets"
export HF_HOME="/work/hm235/hf_cache"

# 设置环境变量
export WANDB_PROJECT=gpt2_frozen_comprehensive

# 运行实验
python experiment_llm.py \\
    --from_scratch \\
    --model_size ${MODEL_SIZE} \\
    --weight_frozen ${weight_frozen} \\
    --qkv_identity_init ${qkv_init} \\
    --identity_alpha ${alpha} \\
    --identity_beta ${beta} \\
    --seed ${seed} \\
    --train_samples ${TRAIN_SAMPLES} \\
    --per_device_train_batch_size ${BATCH_SIZE} \\
    --gradient_accumulation_steps ${GRAD_ACCUM} \\
    --per_device_eval_batch_size 4 \\
    --learning_rate 5e-4 \\
    --weight_decay 0.01 \\
    --max_steps 80000 \\
    --eval_nq_samples 1000 \\
    --eval_lambada_samples 1000 \\
    --eval_wikitext_samples 1000 \\
    --max_length 1024 \\
    --run_name ${job_name} \\
    --output_dir ${OUTPUT_DIR}/${exp_name} \\
    ${freeze_qkv_param} \\
    ${qkv_init_param} \\
    ${skip_flag}

echo "Job ${job_name} completed at \$(date)"
EOF
        
        sleep 0.5  # 避免提交过快
    done
    
    echo ""
done

echo "========================================"
echo "All experiments submitted!"
echo "========================================"
echo ""
echo "Experiment Summary:"
echo "  - Baseline experiments: 3"
echo "  - Freeze Q+K experiments: 3"
echo "  - Freeze K+V experiments: 2"
echo "  - Freeze Q+V experiments: 2"
echo "  - Freeze Q only: 2"
echo "  - Freeze K only: 2"
echo "  - Freeze V only: 2"
echo "  - Differential init experiments: 3"
echo "  - Shallow experiments: 1"
echo "  Total configurations: ${#EXPERIMENTS[@]}"
echo "  Total jobs (with seeds): $((${#EXPERIMENTS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: ${LOGS_DIR}"
echo "View results in wandb project: gpt2_qkv_control"
