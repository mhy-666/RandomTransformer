#!/usr/bin/env python
# coding: utf-8

"""
Continual Learning with SVD Projection: PW + (I-P)R
Stage 1: Train on WikiText-103 with W_eff = PW + (I-P)R, resample R each batch
Stage 2: Freeze PW, train R only on TinyStories
where P = UU^T from SVD of W
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import random
import numpy as np
import wandb
import os
import torch.nn as nn
from tqdm.auto import tqdm
import argparse
import json
from itertools import chain
from transformers.testing_utils import CaptureLogger
import transformers
import logging
from typing import Optional

# 创建logger
logger = logging.getLogger(__name__)

# ============================================================
# SVDProjectedLinear Layer
# ============================================================

class SVDProjectedLinear(nn.Module):
    """
    实现 W_eff = P*W + (I-P)*R 的线性层
    其中 P = U*U^T 来自W的SVD分解

    Stage 1: 训练W和R，每个batch重新采样R
    Stage 2: 冻结W，只训练R
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        original_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        rank: int,
        stage: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.stage = stage

        # SVD分解原始权重
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(original_weight.float(), full_matrices=False)
            # 取前rank个奇异向量
            self.register_buffer('U', U[:, :rank].to(original_weight.dtype))  # [out_features, rank]

        # W参数：用原始权重初始化
        self.W = nn.Parameter(original_weight.clone())

        # R参数：随机初始化
        self.R = nn.Parameter(torch.randn_like(original_weight) * 0.02)

        # Bias
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)

        # Stage 2: 冻结W
        if stage == 2:
            self.W.requires_grad = False
            print(f"  [SVDProjectedLinear] Stage 2: W frozen, R trainable")

    def compute_effective_weight(self, resample_R: bool = False):
        """
        计算有效权重: W_eff = P*W + (I-P)*R
        高效实现: W_eff = P*W + R - P*R = P*(W-R) + R
        其中 P*X = U @ (U^T @ X)
        """
        if resample_R and self.stage == 1 and self.training:
            # Stage 1每个batch重新采样R
            with torch.no_grad():
                self.R.data = torch.randn_like(self.R) * 0.02

        # 高效计算: P*W = U @ (U^T @ W)
        PW = self.U @ (self.U.T @ self.W)

        # (I-P)*R = R - P*R
        PR = self.U @ (self.U.T @ self.R)
        I_minus_P_R = self.R - PR

        W_eff = PW + I_minus_P_R
        return W_eff

    def forward(self, x: torch.Tensor):
        W_eff = self.compute_effective_weight(
            resample_R=(self.stage == 1 and self.training)
        )
        return torch.nn.functional.linear(x, W_eff, self.bias)

    def set_stage(self, stage: int):
        """切换训练阶段"""
        self.stage = stage
        if stage == 2:
            self.W.requires_grad = False
        else:
            self.W.requires_grad = True


def replace_linear_with_svd_projected(
    model: nn.Module,
    rank: int,
    stage: int,
    layer_patterns: Optional[list] = None
):
    """
    替换模型中的Linear层为SVDProjectedLinear

    Args:
        model: 要修改的模型
        rank: SVD的秩
        stage: 1=Stage1训练, 2=Stage2训练
        layer_patterns: 要替换的层名称模式列表，None表示替换所有
    """
    replaced_count = 0

    def should_replace(name):
        # 跳过不需要处理的层
        skip_patterns = ['lm_head', 'wte', 'wpe', 'ln', 'layernorm']
        if any(skip in name.lower() for skip in skip_patterns):
            return False

        # 如果指定了模式，检查是否匹配
        if layer_patterns is not None:
            return any(pattern in name for pattern in layer_patterns)

        return True

    # 收集需要替换的层
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_replace(name):
            layers_to_replace.append((name, module))

    # 执行替换
    for name, module in layers_to_replace:
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        # 创建新的SVD投影层
        svd_linear = SVDProjectedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            original_weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            rank=rank,
            stage=stage
        )

        # 替换
        setattr(parent, child_name, svd_linear)
        replaced_count += 1
        print(f"  ✓ Replaced {name} [{module.weight.shape}] with SVDProjectedLinear (rank={rank})")

    print(f"\n  Total replaced: {replaced_count} layers\n")
    return replaced_count


def set_model_stage(model: nn.Module, stage: int):
    """切换模型中所有SVDProjectedLinear层的训练阶段"""
    count = 0
    for module in model.modules():
        if isinstance(module, SVDProjectedLinear):
            module.set_stage(stage)
            count += 1
    print(f"  Set {count} SVDProjectedLinear layers to stage {stage}")


# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser()

# 实验模式
parser.add_argument('--experiment_mode', type=str, default='two_stage',
                    choices=['stage1_only', 'stage2_only', 'two_stage'],
                    help='stage1_only: 只运行Stage1(WikiText), stage2_only: 只运行Stage2(TinyStories), two_stage: 运行两个阶段')
parser.add_argument('--from_scratch', action='store_true', 
                   help='Initialize model from scratch')
# 模型和路径
parser.add_argument('--model_size', type=str, default='gpt2',
                    choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
parser.add_argument('--load_pretrained_path', type=str, default=None,
                    help='Path to load pretrained model checkpoint from previous stage')
parser.add_argument('--stage1_model_path', type=str, default=None,
                    help='Stage 1 pretrained model path (for stage2_only mode)')
parser.add_argument('--output_dir', type=str, default='./svd_continual_output')
parser.add_argument('--project_name', type=str, default='svd_continual_learning')
parser.add_argument('--run_name', type=str, default=None)

# SVD配置
parser.add_argument('--svd_rank', type=int, default=256,
                    help='SVD rank for projection P=UU^T')
parser.add_argument('--apply_svd_to', type=str, default='all',
                    choices=['all', 'attn', 'mlp'],
                    help='Which layers to apply SVD projection')

# 训练参数 - Stage 1 (WikiText-103)
parser.add_argument('--stage1_per_device_train_batch_size', type=int, default=4)
parser.add_argument('--stage1_gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--stage1_learning_rate', type=float, default=5e-4)
parser.add_argument('--stage1_weight_decay', type=float, default=0.01)
parser.add_argument('--stage1_warmup_steps', type=int, default=500)
parser.add_argument('--stage1_max_steps', type=int, default=10000)
parser.add_argument('--stage1_save_steps', type=int, default=2000)
parser.add_argument('--stage1_logging_steps', type=int, default=50)
parser.add_argument('--stage1_eval_steps', type=int, default=1000)

# 训练参数 - Stage 2 (TinyStories)
parser.add_argument('--stage2_per_device_train_batch_size', type=int, default=4)
parser.add_argument('--stage2_gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--stage2_learning_rate', type=float, default=5e-5)
parser.add_argument('--stage2_weight_decay', type=float, default=0.01)
parser.add_argument('--stage2_warmup_steps', type=int, default=500)
parser.add_argument('--stage2_max_steps', type=int, default=10000)
parser.add_argument('--stage2_save_steps', type=int, default=2000)
parser.add_argument('--stage2_logging_steps', type=int, default=50)
parser.add_argument('--stage2_eval_steps', type=int, default=1000)

# 数据参数
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--max_train_samples', type=int, default=None)

# 其他
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()

# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ============================================================
# WandB初始化
# ============================================================
run_name = args.run_name or f"svd_r{args.svd_rank}_{args.apply_svd_to}_{args.experiment_mode}"
wandb.init(project=args.project_name, name=run_name, config=vars(args))

# ============================================================
# 数据加载函数
# ============================================================

def load_and_preprocess_wikitext(tokenizer, max_length, max_samples=None):
    """加载和预处理WikiText-103"""
    print("\n" + "=" * 70)
    print("Loading WikiText-103 dataset...")
    print("=" * 70)

    raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1", keep_in_memory=True)

    if max_samples:
        raw_datasets["train"] = raw_datasets["train"].select(range(max_samples))

    print(f"✓ Train samples: {len(raw_datasets['train']):,}")
    print(f"✓ Validation samples: {len(raw_datasets['validation']):,}")

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text"

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing WikiText-103",
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_length}",
    )

    print(f"✓ Tokenized train blocks: {len(lm_datasets['train']):,}")
    print(f"✓ Tokenized validation blocks: {len(lm_datasets['validation']):,}")

    return lm_datasets["train"], lm_datasets["validation"]


def load_and_preprocess_tinystories(tokenizer, max_length, max_samples=None):
    """加载和预处理TinyStories"""
    print("\n" + "=" * 70)
    print("Loading TinyStories dataset...")
    print("=" * 70)

    raw_datasets = load_dataset("roneneldan/TinyStories", keep_in_memory=True)

    if max_samples:
        raw_datasets["train"] = raw_datasets["train"].select(range(max_samples))

    print(f"✓ Train samples: {len(raw_datasets['train']):,}")
    print(f"✓ Validation samples: {len(raw_datasets['validation']):,}")

    column_names = list(raw_datasets["train"].features)
    text_column_name = "text"

    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing TinyStories",
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {max_length}",
    )

    print(f"✓ Tokenized train blocks: {len(lm_datasets['train']):,}")
    print(f"✓ Tokenized validation blocks: {len(lm_datasets['validation']):,}")

    return lm_datasets["train"], lm_datasets["validation"]


# ============================================================
# 评估函数
# ============================================================

@torch.no_grad()
def evaluate_wikitext103(model, tokenizer, device, max_length=512):
    """评估WikiText-103"""
    print("\n" + "=" * 70)
    print("Evaluating WikiText-103...")
    print("=" * 70)
    model.eval()

    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", keep_in_memory=True)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    stride = max_length
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="WikiText-103 Eval"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)

    print(f"✓ WikiText-103 PPL: {ppl:.2f}")
    print("=" * 70)

    model.train()
    return ppl.item()


@torch.no_grad()
def evaluate_tinystories(model, tokenizer, device, max_length=512):
    """评估TinyStories"""
    print("\n" + "=" * 70)
    print("Evaluating TinyStories...")
    print("=" * 70)
    model.eval()

    test_data = load_dataset("roneneldan/TinyStories", split="validation", keep_in_memory=True)
    all_text = "\n\n".join(test_data["text"][:1000])
    encodings = tokenizer(all_text, return_tensors="pt")

    stride = max_length
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="TinyStories Eval"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)

    print(f"✓ TinyStories PPL: {ppl:.2f}")
    print("=" * 70)

    model.train()
    return ppl.item()


# ============================================================
# Callback
# ============================================================

class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            try:
                perplexity = np.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["eval_perplexity"] = perplexity
            print(f"Evaluation metrics: {metrics}")
        return control


# ============================================================
# 主训练流程
# ============================================================

print("\n" + "=" * 70)
print("SVD Continual Learning Experiment")
print(f"Mode: {args.experiment_mode}")
print(f"SVD Rank: {args.svd_rank}")
print(f"Apply to: {args.apply_svd_to}")
print("=" * 70)

# 加载tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# STAGE 1: WikiText-103
# ============================================================

if args.experiment_mode in ['stage1_only', 'two_stage']:
    print("\n" + "=" * 70)
    print("STAGE 1: Training on WikiText-103")
    print("=" * 70)

    # 加载模型
    print(f"\n[Step 1] Loading/Initializing GPT-2 model ({args.model_size})...")
    if args.load_pretrained_path:
        print("=" * 70)
        print(f"Loading pretrained model from: {args.load_pretrained_path}")
        print("=" * 70)
        model = GPT2LMHeadModel.from_pretrained(args.load_pretrained_path)
        print(f"✓ Successfully loaded model")
        model.to(args.device)
    elif args.from_scratch:
        # config = GPT2Config.from_pretrained(args.model_size)
        config = GPT2Config(
            vocab_size=50257,
            n_positions=args.max_length,  # 使用你参数里的 max_length
            n_ctx=args.max_length,
            n_embd=768,           # 关键修改
            n_layer=12,            # 关键修改
            n_head=12,             # 关键修改
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # resid_pdrop=0.35,  # 残差连接 Dropout
            # embd_pdrop=0.35,   # Embedding Dropout
            # attn_pdrop=0.35,   # Attention Dropout
        )
        model = GPT2LMHeadModel(config)
        print(f"✓ Initialized {args.model_size} from scratch")
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_size)
        print(f"✓ Loaded pretrained {args.model_size}")

    model = model.to(args.device)

    # 应用SVD投影
    print(f"\n[Applying SVD Projection - Stage 1]")
    print(f"  Rank: {args.svd_rank}")
    print(f"  Target layers: {args.apply_svd_to}")

    if args.apply_svd_to == 'all':
        layer_patterns = ['attn', 'mlp']
    elif args.apply_svd_to == 'attn':
        layer_patterns = ['attn']
    elif args.apply_svd_to == 'mlp':
        layer_patterns = ['mlp']

    replace_linear_with_svd_projected(
        model,
        rank=args.svd_rank,
        stage=1,
        layer_patterns=layer_patterns
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 加载数据
    train_dataset_wiki, val_dataset_wiki = load_and_preprocess_wikitext(
        tokenizer,
        args.max_length,
        max_samples=args.max_train_samples
    )

    # 评估初始性能
    print("\n[Stage 1 - Before Training]")
    wiki_ppl_s1_before = evaluate_wikitext103(model, tokenizer, args.device, args.max_length)
    tiny_ppl_s1_before = evaluate_tinystories(model, tokenizer, args.device, args.max_length)

    wandb.log({'stage1/before/wikitext_ppl': wiki_ppl_s1_before})
    wandb.log({'stage1/before/tinystory_ppl': tiny_ppl_s1_before})

    # 训练
    output_dir_stage1 = f"{args.output_dir}/stage1_{run_name}"
    os.makedirs(output_dir_stage1, exist_ok=True)

    training_args_stage1 = TrainingArguments(
        output_dir=output_dir_stage1,
        overwrite_output_dir=True,
        max_steps=args.stage1_max_steps,
        per_device_train_batch_size=args.stage1_per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.stage1_gradient_accumulation_steps,
        save_steps=args.stage1_save_steps,
        save_total_limit=2,
        weight_decay=args.stage1_weight_decay,
        warmup_steps=args.stage1_warmup_steps,
        learning_rate=args.stage1_learning_rate,
        logging_steps=args.stage1_logging_steps,
        logging_dir=f'{output_dir_stage1}/logs',
        report_to=['wandb'],
        eval_strategy="steps",
        eval_steps=args.stage1_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
    )

    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=train_dataset_wiki,
        eval_dataset=val_dataset_wiki,
        callbacks=[EvalCallback()],
    )

    print("\n[Starting Stage 1 Training...]")
    trainer_stage1.train()

    # 评估训练后性能
    print("\n[Stage 1 - After Training]")
    wiki_ppl_s1_after = evaluate_wikitext103(model, tokenizer, args.device, args.max_length)

    # 保存模型
    trainer_stage1.save_model(f"{output_dir_stage1}/final_model")
    print(f"\n✓ Stage 1 model saved to {output_dir_stage1}/final_model")

    # 保存结果
    stage1_results = {
        'stage': 1,
        'dataset': 'WikiText-103',
        'svd_rank': args.svd_rank,
        'before_training': {'wikitext_ppl': wiki_ppl_s1_before},
        'after_training': {'wikitext_ppl': wiki_ppl_s1_after},
        'improvement': wiki_ppl_s1_before - wiki_ppl_s1_after,
    }

    with open(f"{output_dir_stage1}/stage1_results.json", 'w') as f:
        json.dump(stage1_results, f, indent=2)

    wandb.log({
        'stage1/after/wikitext_ppl': wiki_ppl_s1_after,
        'stage1/improvement': wiki_ppl_s1_before - wiki_ppl_s1_after,
    })

    # 如果是two_stage模式，保存模型路径供Stage 2使用
    if args.experiment_mode == 'two_stage':
        stage1_model_path = f"{output_dir_stage1}/final_model"


# ============================================================
# STAGE 2: TinyStories
# ============================================================

if args.experiment_mode in ['stage2_only', 'two_stage']:
    print("\n" + "=" * 70)
    print("STAGE 2: Training on TinyStories")
    print("=" * 70)

    # 加载Stage 1模型
    if args.experiment_mode == 'stage2_only':
        if args.stage1_model_path is None:
            raise ValueError("--stage1_model_path must be provided for stage2_only mode")
        stage1_model_path = args.stage1_model_path

    print(f"\nLoading Stage 1 model from: {stage1_model_path}")
    model = GPT2LMHeadModel.from_pretrained(stage1_model_path)
    model.to(args.device)

    # 切换到Stage 2（冻结PW，只训练R）
    print("\n[Switching to Stage 2: Freezing PW, training R only]")
    set_model_stage(model, stage=2)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params: {total_params:,}")
    print(f"  Trainable params (R only): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 加载TinyStories数据
    train_dataset_tiny, val_dataset_tiny = load_and_preprocess_tinystories(
        tokenizer,
        args.max_length,
        max_samples=args.max_train_samples
    )

    # 评估初始性能（Stage 1模型在两个数据集上的表现）
    print("\n[Stage 2 - Before Training]")
    tiny_ppl_s2_before = evaluate_tinystories(model, tokenizer, args.device, args.max_length)
    wiki_ppl_s2_before = evaluate_wikitext103(model, tokenizer, args.device, args.max_length)

    wandb.log({
        'stage2/before/tinystories_ppl': tiny_ppl_s2_before,
        'stage2/before/wikitext_ppl': wiki_ppl_s2_before,
    })

    # 训练
    output_dir_stage2 = f"{args.output_dir}/stage2_{run_name}"
    os.makedirs(output_dir_stage2, exist_ok=True)

    training_args_stage2 = TrainingArguments(
        output_dir=output_dir_stage2,
        overwrite_output_dir=True,
        max_steps=args.stage2_max_steps,
        per_device_train_batch_size=args.stage2_per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.stage2_gradient_accumulation_steps,
        save_steps=args.stage2_save_steps,
        save_total_limit=22,
        weight_decay=args.stage2_weight_decay,
        warmup_steps=args.stage2_warmup_steps,
        learning_rate=args.stage2_learning_rate,
        logging_steps=args.stage2_logging_steps,
        logging_dir=f'{output_dir_stage2}/logs',
        report_to=['wandb'],
        eval_strategy="steps",
        eval_steps=args.stage2_eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
    )

    trainer_stage2 = Trainer(
        model=model,
        args=training_args_stage2,
        train_dataset=train_dataset_tiny,
        eval_dataset=val_dataset_tiny,
        callbacks=[EvalCallback()],
    )

    print("\n[Starting Stage 2 Training...]")
    trainer_stage2.train()

    # 评估训练后性能
    print("\n[Stage 2 - After Training]")
    tiny_ppl_s2_after = evaluate_tinystories(model, tokenizer, args.device, args.max_length)
    wiki_ppl_s2_after = evaluate_wikitext103(model, tokenizer, args.device, args.max_length)

    # 保存模型
    trainer_stage2.save_model(f"{output_dir_stage2}/final_model")
    print(f"\n✓ Stage 2 model saved to {output_dir_stage2}/final_model")

    # 计算性能变化
    tiny_improvement = tiny_ppl_s2_before - tiny_ppl_s2_after
    tiny_improvement_pct = (tiny_improvement / tiny_ppl_s2_before) * 100

    wiki_change = wiki_ppl_s2_after - wiki_ppl_s2_before
    forgetting_pct = (wiki_change / wiki_ppl_s2_before) * 100

    # 打印结果对比
    print("\n" + "=" * 70)
    print("STAGE 2 - Performance Comparison")
    print("=" * 70)
    print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'TinyStories PPL':<30} {tiny_ppl_s2_before:<15.2f} {tiny_ppl_s2_after:<15.2f} {tiny_improvement:+.2f} ({tiny_improvement_pct:+.1f}%)")
    print(f"{'WikiText-103 PPL':<30} {wiki_ppl_s2_before:<15.2f} {wiki_ppl_s2_after:<15.2f} {wiki_change:+.2f} ({forgetting_pct:+.2f}%)")
    print("=" * 70)

    # 保存结果
    stage2_results = {
        'stage': 2,
        'dataset': 'TinyStories',
        'svd_rank': args.svd_rank,
        'before_training': {
            'tinystories_ppl': tiny_ppl_s2_before,
            'wikitext_ppl': wiki_ppl_s2_before,
        },
        'after_training': {
            'tinystories_ppl': tiny_ppl_s2_after,
            'wikitext_ppl': wiki_ppl_s2_after,
        },
        'performance_change': {
            'tinystories_improvement': tiny_improvement,
            'tinystories_improvement_pct': tiny_improvement_pct,
            'wikitext_change': wiki_change,
            'forgetting_pct': forgetting_pct,
        },
    }

    with open(f"{output_dir_stage2}/stage2_results.json", 'w') as f:
        json.dump(stage2_results, f, indent=2)

    wandb.log({
        'stage2/after/tinystories_ppl': tiny_ppl_s2_after,
        'stage2/after/wikitext_ppl': wiki_ppl_s2_after,
        'stage2/tinystories_improvement': tiny_improvement,
        'stage2/tinystories_improvement_pct': tiny_improvement_pct,
        'stage2/wikitext_change': wiki_change,
        'stage2/forgetting_pct': forgetting_pct,
    })

wandb.finish()

print("\n" + "=" * 70)
print("✓ Experiment completed!")
print("=" * 70)
