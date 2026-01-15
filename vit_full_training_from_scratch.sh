#!/bin/bash

# 定义实验配置
SEEDS=(4) 
MODEL="tiny_vit_21m_224.in1k"
BATCH_SIZE=128
EPOCHS=100
OUTPUT_DIR="/work/hm235/random_transformer/outputs/tinyvit_experiments"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# ImageNet 数据路径
DATA_DIR="/work/hm235/random_transformer/data/hf_cache/"

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3" 
CONDA_ENV="tinyvit"

# 实验配置列表
declare -A EXPERIMENTS

# ========== 基线实验 ==========

EXPERIMENTS[-1]="baseline_zeroshot:-1:none:0.8:0.2:none:::Zero-shot pretrained model (no training)"
# EXPERIMENTS[0]="baseline_0_full_finetune:0:none:0.8:0.2:none:::Full finetune all parameters"
# EXPERIMENTS[1]="baseline_1_freeze_attn_mlp:1:none:0.8:0.2:none:::Freeze Attention + MLP in all TinyVitBlocks"
# EXPERIMENTS[2]="baseline_2_freeze_attn:2:none:0.8:0.2:none:::Freeze Attention only in all TinyVitBlocks"
# EXPERIMENTS[3]="baseline_3_freeze_mlp:3:none:0.8:0.2:none:::Freeze MLP only in all TinyVitBlocks"
# EXPERIMENTS[4]="baseline_4_freeze_attn_mlp_patch:4:none:0.8:0.2:none:::Freeze Attention + MLP + PatchEmbed"
echo "========================================"
echo "Submitting TinyViT Fine-grained Control Experiments"
echo "========================================"
echo ""

# 遍历所有实验配置
for exp_key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name weight_frozen qkv_init alpha beta freeze_qkv_comp freeze_stages freeze_blocks description <<< "${EXPERIMENTS[$exp_key]}"
    
    echo "Experiment $exp_key: $description"
    echo "  Freeze strategy: weight_frozen=$weight_frozen"
    echo "  QKV components to freeze: $freeze_qkv_comp"
    echo "  Freeze stages: $freeze_stages"
    echo "  Freeze blocks: $freeze_blocks"
    echo "  QKV init strategy: $qkv_init (alpha=$alpha, beta=$beta)"
    
    # 零样本实验不需要设置epochs
    if [ "$weight_frozen" = "-1" ]; then
        ACTUAL_EPOCHS=1
        echo "  Mode: Zero-shot evaluation (no training)"
    else
        ACTUAL_EPOCHS=${EPOCHS}
        echo "  Mode: Training for ${ACTUAL_EPOCHS} epochs"
    fi

    # 遍历所有 seeds
    for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        
        echo "  - Submitting job: $job_name"
        
        # 构建freeze_qkv参数
        if [ "$freeze_qkv_comp" = "none" ]; then
            freeze_qkv_param=""
        else
            freeze_qkv_param="--freeze_qkv_components ${freeze_qkv_comp}"
        fi
        
        # 构建freeze_stages参数
        if [ -z "$freeze_stages" ] || [ "$freeze_stages" = "none" ]; then
            freeze_stages_param=""
        else
            freeze_stages_param="--freeze_stages ${freeze_stages}"
        fi
        
        # 构建freeze_blocks参数
        if [ -z "$freeze_blocks" ] || [ "$freeze_blocks" = "none" ]; then
            freeze_blocks_param=""
        else
            freeze_blocks_param="--freeze_blocks ${freeze_blocks}"
        fi
        
        # 创建临时作业脚本
        temp_script=$(mktemp /tmp/${job_name}_XXXXXX.sbatch)
        
        # 将作业脚本内容写入临时文件
        cat > ${temp_script} <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOGS_DIR}/${job_name}_%j.out
#SBATCH --error=${LOGS_DIR}/${job_name}_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-common

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

echo "========================================"
echo "🎯 Job: ${job_name}"
echo "========================================"
echo "Node: \$(hostname)"
echo "Start time: \$(date)"
echo "Working directory: \$(pwd)"
echo ""
echo "Environment:"
echo "  Conda env: \${CONDA_DEFAULT_ENV}"
echo "  Python: \$(python --version)"
echo "  Python path: \$(which python)"
echo ""

# 测试PyTorch
echo "Testing PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPU count: {torch.cuda.device_count()}')"

if [ \$? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: PyTorch import failed!"
    exit 1
fi

echo ""
echo "✓ Environment check passed"
echo "========================================"
echo ""
echo "Starting training..."
echo ""

python experiment_vit.py --model ${MODEL} --data_dir ${DATA_DIR} --weight_frozen ${weight_frozen} --qkv_identity_init ${qkv_init} --identity_alpha ${alpha} --identity_beta ${beta} --seed ${seed} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --workers 8 --lr 5e-4 --weight_decay 0.05 --opt adamw --sched cosine --warmup_epochs 5 --warmup_lr 1e-6 --min_lr 1e-5 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --reprob 0.25 --drop_path 0.1 --run_name ${job_name} --output_dir ${OUTPUT_DIR}/${exp_name} --log_interval 100 --eval_interval 5 ${freeze_qkv_param} ${freeze_stages_param} ${freeze_blocks_param}

EXIT_CODE=\$?

echo ""
echo "======================================"
echo "Job ${job_name} completed"
echo "End time: \$(date)"
echo "Exit code: \$EXIT_CODE"
echo "======================================"

exit \$EXIT_CODE
EOF
        
        # 提交作业
        sbatch ${temp_script}
        
        # 删除临时文件
        rm ${temp_script}
        
        sleep 0.5
    done
    
    echo ""
done

echo "========================================"
echo "All experiments submitted!"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOGS_DIR}/<job_name>_<jobid>.out"
