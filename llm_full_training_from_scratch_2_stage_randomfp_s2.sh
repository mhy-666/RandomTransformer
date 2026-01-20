#!/bin/bash

# ============================================================================
# Stage 2: 全参数训练实验
# 从Stage 1的6个checkpoint继续训练
# 每个checkpoint只测试：继续相同的random FP 或 切换到standard training
# ============================================================================

nvidia-smi

# 基础配置
SEEDS=(42)
MODEL_SIZE="gpt2"
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_STEPS=40000
LEARNING_RATE=5e-4

# 目录配置
BASE_DIR="/work/hm235/random_transformer"
STAGE1_OUTPUT_DIR="${BASE_DIR}/outputs/random_forward"
OUTPUT_DIR="${BASE_DIR}/outputs/random_forward_stage2"
LOGS_DIR="${OUTPUT_DIR}/logs"
mkdir -p ${LOGS_DIR}

# Python脚本所在目录
WORK_DIR="/hpc/home/hm235/Desktop/random_transformers"

# Conda路径
CONDA_PATH="/work/hm235/miniconda3"
CONDA_ENV="tinyvit"

# 实验配置: stage1_exp:s2_random_fp:description
declare -A EXPERIMENTS

# 从 freeze_attn_rfp_attn 继续
EXPERIMENTS[1]="freeze_attn_rfb_attn:attn:S1(freeze Attn, RFP on Attn) -> S2(full train, RFP on Attn)"
EXPERIMENTS[2]="freeze_attn_rfb_attn:none:S1(freeze Attn, RFP on Attn) -> S2(full train, standard)"

# 从 freeze_mlp_rfp_mlp 继续
EXPERIMENTS[3]="freeze_mlp_rfb_mlp:mlp:S1(freeze MLP, RFP on MLP) -> S2(full train, RFP on MLP)"
EXPERIMENTS[4]="freeze_mlp_rfb_mlp:none:S1(freeze MLP, RFP on MLP) -> S2(full train, standard)"

# 从 freeze_both_rfp_both 继续
EXPERIMENTS[5]="freeze_both_rfb_both:attn_mlp:S1(freeze Both, RFP on Both) -> S2(full train, RFP on Both)"
EXPERIMENTS[6]="freeze_both_rfb_both:none:S1(freeze Both, RFP on Both) -> S2(full train, standard)"

echo "========================================================================"
echo "Stage 2: Full Training - Continue Random FP vs Standard Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Max steps: ${MAX_STEPS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Strategy: weight_frozen=0 (FULL TRAINING)"
echo ""
echo "Stage 1 checkpoints: 6 (3 freeze strategies × 2 modes: fixed/resample)"
echo "Stage 2 configs per checkpoint: 3"
echo "  - Continue with same random FP (fixed mode)"
echo "  - Continue with same random FP (resample mode)"
echo "  - Switch to standard training"
echo ""
echo "Total experiments: 6 checkpoints × 3 configs = 18 experiments"
echo ""

exp_count=0

# 遍历所有实验配置
for exp_key in "${!EXPERIMENTS[@]}"; do
  IFS=':' read -r stage1_base s2_random_fp base_description <<< "${EXPERIMENTS[$exp_key]}"

  # 遍历Stage 1的fixed和resample两个版本
  for stage1_mode in "fixed" "resample"; do
    stage1_exp="${stage1_base}_${stage1_mode}"

    # 如果是standard training，只运行一次（不分fixed/resample）
    if [ "$s2_random_fp" = "none" ]; then
      modes=("standard")
    else
      # 如果继续用random FP，测试fixed和resample
      modes=("fixed" "resample")
    fi

    for s2_mode in "${modes[@]}"; do
      if [ "$s2_mode" = "standard" ]; then
        exp_name="s2_from_${stage1_exp}_to_standard"
        resample_flag=""
        mode_desc="standard training (no random FP)"
      elif [ "$s2_mode" = "resample" ]; then
        exp_name="s2_from_${stage1_exp}_to_${s2_random_fp}_resample"
        resample_flag="--resample_every_batch"
        mode_desc="continue RFP on ${s2_random_fp} (resample)"
      else
        exp_name="s2_from_${stage1_exp}_to_${s2_random_fp}_fixed"
        resample_flag=""
        mode_desc="continue RFP on ${s2_random_fp} (fixed)"
      fi

      exp_count=$((exp_count + 1))

      echo "======================================================================"
      echo "Experiment ${exp_count}: ${base_description}"
      echo "======================================================================"
      echo "  Stage 1: ${stage1_exp}"
      echo "  Stage 2: ${mode_desc}"
      echo ""

      for seed in "${SEEDS[@]}"; do
        job_name="${exp_name}_seed${seed}"
        output_dir="${OUTPUT_DIR}/${exp_name}_seed${seed}"

        # Stage 1 checkpoint路径
        stage1_checkpoint="${STAGE1_OUTPUT_DIR}/${stage1_exp}_seed${seed}/${stage1_exp}_seed${seed}/final_model"

        # 检查Stage 1 checkpoint是否存在
        if [ ! -d "${stage1_checkpoint}" ]; then
          echo "  ⚠️  WARNING: Stage 1 checkpoint not found: ${stage1_checkpoint}"
          echo "  Skipping: $job_name"
          echo ""
          continue
        fi

        echo "  Submitting: $job_name"
        echo "  Loading from: ${stage1_checkpoint}"

        # 构建random FP参数
        if [ "$s2_random_fp" = "none" ]; then
          random_fp_params=""
        else
          random_fp_params="--random_forward_strategy full_random"
          random_fp_params="${random_fp_params} --apply_random_forward_to_layers ${s2_random_fp}"
          random_fp_params="${random_fp_params} --projection_type random"
          random_fp_params="${random_fp_params} ${resample_flag}"
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
#SBATCH --time=7:00:00
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
    --weight_frozen 0 \
    --load_pretrained_path ${stage1_checkpoint} \
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
    --project_name "random_forward_stage2" \
    --run_name "${job_name}" \
    ${random_fp_params}

echo "Job ${job_name} completed!"
EOF

        echo ""
      done
    done
  done
done

echo "========================================================================"
echo "All jobs submitted!"
echo "========================================================================"
