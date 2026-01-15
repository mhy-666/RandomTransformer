#!/usr/bin/env python
# coding: utf-8

"""
Continual Learning on TinyStories
基于WikiText-103预训练模型，使用与原代码相同的架构和训练流程
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

# 创建logger
logger = logging.getLogger(__name__)

# ============================================================
# 参数解析
# ============================================================

parser = argparse.ArgumentParser()

# 模型和路径
parser.add_argument('--load_pretrained_path', type=str, required=True,
                    help='Path to WikiText-103 pretrained model')
parser.add_argument('--output_dir', type=str, default='./tinystories_continual_output')
parser.add_argument('--project_name', type=str, default='gpt2_tinystories_continual')
parser.add_argument('--run_name', type=str, default=None)

# Freeze策略 (使用你原代码的策略系统)
parser.add_argument('--weight_frozen', type=int, default=297,
                    help='297: train MLP only; 298: train attn only; 2: freeze attn only; 3: freeze mlp only')
parser.add_argument('--freeze_qkv_components', type=str, default='all',
                    choices=['all', 'none', 'q', 'k', 'v', 'qk', 'qv', 'kv'])

# 训练参数
parser.add_argument('--per_device_train_batch_size', type=int, default=4)
parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--max_steps', type=int, default=10000)
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--eval_steps', type=int, default=1000)

# 数据参数
parser.add_argument('--max_train_samples', type=int, default=None,
                    help='Max training samples from TinyStories (None=all)')

# 其他
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--skip_training', action='store_true')

args = parser.parse_args()

# ============================================================
# 工具函数 (从你的代码中复制)
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# Freeze策略名称映射
freeze_strategy_names = {
    -1: "zeroshot",
    0: "full_finetune",
    1: "freeze_attn_mlp",
    2: "freeze_attn_only",
    3: "freeze_mlp_only",
    100: "freeze_attn_mlp_pos",
    297: "train_mlp_only",
    298: "train_attn_only",
    299: "train_attn_mlp",
    200: "freeze_all"
}

def get_strategy_description(strategy):
    """获取策略描述"""
    descriptions = {
        297: {'name': 'Train MLP Only', 'frozen': 'Attention + Embeddings', 'trainable': 'MLP'},
        298: {'name': 'Train Attention Only', 'frozen': 'MLP + Embeddings', 'trainable': 'Attention'},
        2: {'name': 'Freeze Attention', 'frozen': 'Attention', 'trainable': 'MLP + Embeddings'},
        3: {'name': 'Freeze MLP', 'frozen': 'MLP', 'trainable': 'Attention + Embeddings'},
    }
    return descriptions.get(strategy, {'name': 'Custom', 'frozen': 'Custom', 'trainable': 'Custom'})

def parse_qkv_components(component_str):
    """解析QKV组件字符串"""
    if component_str == 'all':
        return ['q', 'k', 'v']
    elif component_str == 'none':
        return []
    else:
        components = []
        if 'q' in component_str:
            components.append('q')
        if 'k' in component_str:
            components.append('k')
        if 'v' in component_str:
            components.append('v')
        return components

def apply_freeze_strategy(model, strategy, freeze_qkv_components='all'):
    """应用冻结策略 (与你的原代码相同)"""
    print("=" * 70)
    print(f"Determining freeze strategy: {strategy}")
    print(f"QKV component control: {freeze_qkv_components}")
    print("=" * 70)
    
    layers_to_freeze = []
    qkv_components = parse_qkv_components(freeze_qkv_components)
    
    strategies_freeze_attn = [1, 2, 100, 297, -1]
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # Attention层处理
        if 'attn' in name:
            if strategy in strategies_freeze_attn:
                if 'attn.c_attn' in name or 'attn.c_proj' in name:
                    should_freeze = True
                else:
                    should_freeze = True
        
        # 策略逻辑
        if strategy == -1:
            should_freeze = True
        elif strategy == 0:
            should_freeze = False
        elif strategy == 1:
            if 'mlp' in name:
                should_freeze = True
        elif strategy == 2:
            pass  # Attention已处理
        elif strategy == 3:
            if 'mlp' in name:
                should_freeze = True
        elif strategy == 100:
            if 'mlp' in name or 'wpe' in name:
                should_freeze = True
        elif strategy == 297:  # Train MLP only
            if 'wte' in name or 'wpe' in name or 'lm_head' in name:
                should_freeze = True
        elif strategy == 298:  # Train Attention only
            if 'mlp' in name or 'wte' in name or 'wpe' in name or 'lm_head' in name:
                should_freeze = True
        elif strategy == 299:
            if any(x in name for x in ['wte', 'wpe', 'lm_head']):
                should_freeze = True
        
        if should_freeze:
            layers_to_freeze.append(name)
    
    strategy_info = get_strategy_description(strategy)
    print(f"Strategy {strategy}: {strategy_info['name']}")
    print(f"  Frozen: {strategy_info['frozen']}")
    print(f"  Trainable: {strategy_info['trainable']}")
    print(f"Total parameters to freeze: {len(layers_to_freeze)}")
    print("=" * 70)
    
    return layers_to_freeze, qkv_components

def execute_freeze(model, layers_to_freeze, qkv_components=['q', 'k', 'v']):
    """执行冻结操作"""
    num_frozen = 0
    num_trainable = 0
    
    print("\n" + "=" * 70)
    print("Executing freeze operations...")
    print("=" * 70)
    
    for name, param in model.named_parameters():
        if name in layers_to_freeze:
            param.requires_grad = False
            num_frozen += param.numel()
            print(f"  [FROZEN]    {name:60s}")
        else:
            param.requires_grad = True
            num_trainable += param.numel()
            print(f"  [TRAINABLE] {name:60s} {tuple(param.shape)}")
    
    total_params = num_trainable + num_frozen
    print("=" * 70)
    print(f"Trainable: {num_trainable:,} ({num_trainable/total_params*100:.2f}%)")
    print(f"Frozen:    {num_frozen:,} ({num_frozen/total_params*100:.2f}%)")
    print(f"Total:     {total_params:,}")
    print("=" * 70 + "\n")
    
    return num_trainable, num_frozen

# ============================================================
# WandB初始化
# ============================================================

run_name = args.run_name or f"tinystories_{freeze_strategy_names.get(args.weight_frozen, 'custom')}_lr{args.learning_rate}"
wandb.init(project=args.project_name, name=run_name, config=vars(args))

# ============================================================
# 加载模型和Tokenizer
# ============================================================

print("=" * 70)
print("Loading pretrained model from WikiText-103...")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(args.load_pretrained_path)
model.to(args.device)

print(f"✓ Model loaded: {model.config.n_layer} layers, {model.config.n_embd} hidden dim")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 应用Freeze策略
# ============================================================

print(f"\n[Step 1] Determining freeze strategy...")
layers_to_freeze, qkv_components_to_freeze = apply_freeze_strategy(
    model,
    args.weight_frozen,
    freeze_qkv_components=args.freeze_qkv_components
)

print(f"\n[Step 2] Freezing parameters...")
num_trainable, num_frozen = execute_freeze(model, layers_to_freeze, qkv_components_to_freeze)

wandb.log({
    'num_trainable': num_trainable,
    'num_frozen': num_frozen,
    'trainable_ratio': num_trainable / (num_trainable + num_frozen)
})

# ============================================================
# 加载TinyStories数据集 (使用与WikiText-103相同的处理方式)
# ============================================================

print("\n" + "=" * 70) 
print("Loading TinyStories dataset...")
print("=" * 70)

# 加载数据集
raw_datasets = load_dataset("roneneldan/TinyStories", keep_in_memory=True)

if args.max_train_samples:
    raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

print(f"✓ Train samples: {len(raw_datasets['train']):,}")
print(f"✓ Validation samples: {len(raw_datasets['validation']):,}")
print(f"  Example: {raw_datasets['train'][0]['text'][:200]}...")

# ============================================================
# 数据预处理 (与WikiText-103完全相同的方式)
# ============================================================

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

def tokenize_function(examples):
    """Tokenize函数 (与WikiText-103相同)"""
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output

def group_texts(examples):
    """分组函数 (与WikiText-103相同)"""
    # 拼接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 按block_size分块
    block_size = args.max_length
    total_length = (total_length // block_size) * block_size
    
    # 切分
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("\nTokenizing dataset...")
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)

block_size = min(args.max_length, tokenizer.model_max_length)
print(f"Block size: {block_size}")

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

print(f"✓ Tokenized train blocks: {len(lm_datasets['train']):,}")
print(f"✓ Tokenized validation blocks: {len(lm_datasets['validation']):,}")

train_dataset = lm_datasets["train"]
val_dataset = lm_datasets["validation"]

# ============================================================
# 评估函数 (与WikiText-103完全相同)
# ============================================================

@torch.no_grad()
def evaluate_tinystories(model, tokenizer, device):
    """
    评估TinyStories - 使用与WikiText-103相同的stride-based方法
    """
    print("\n" + "=" * 70)
    print("Evaluating TinyStories...")
    print("=" * 70)
    
    model.eval()
    
    # 加载test split
    test_data = load_dataset("roneneldan/TinyStories", split="validation", keep_in_memory=True)
    
    # 拼接所有文本
    all_text = "\n\n".join(test_data["text"][:1000])  # 使用前1000个样本
    encodings = tokenizer(all_text, return_tensors="pt")
    
    max_length = args.max_length
    stride = args.max_length
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

@torch.no_grad()
def evaluate_wikitext103(model, tokenizer, device):
    """
    评估WikiText-103 - 检查遗忘 (与你的原代码完全相同)
    """
    print("\n" + "=" * 70)
    print("Evaluating WikiText-103 (checking forgetting)...")
    print("=" * 70)
    
    model.eval()
    
    test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", keep_in_memory=True)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    max_length = args.max_length
    stride = args.max_length
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

# ============================================================
# 训练前评估 (Baseline)
# ============================================================

print("\n" + "=" * 70)
print("BEFORE TRAINING - Baseline Evaluation")
print("=" * 70)

tinystories_ppl_before = evaluate_tinystories(model, tokenizer, args.device)
wikitext_ppl_before = evaluate_wikitext103(model, tokenizer, args.device)

wandb.log({
    'before/tinystories_ppl': tinystories_ppl_before,
    'before/wikitext103_ppl': wikitext_ppl_before,
})

# ============================================================
# Trainer训练 (与WikiText-103相同的方式)
# ============================================================

output_dir = f"{args.output_dir}/{run_name}"
os.makedirs(output_dir, exist_ok=True)

# Callback - 计算perplexity
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

if not args.skip_training and args.weight_frozen != -1:
    # 创建optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        logging_dir=f'{output_dir}/logs',
        report_to=['wandb'],
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
    )
    
    # Trainer
    eval_callback = EvalCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[eval_callback],
    )
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    trainer.train()
    
    # 评估
    metrics = trainer.evaluate()
    try:
        perplexity = np.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["eval_perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    # 保存模型
    trainer.save_model(f"{output_dir}/final_model")
    print(f"✓ Model saved to {output_dir}/final_model")

else:
    print("\nSkipping training (zero-shot mode)")

# ============================================================
# 训练后最终评估
# ============================================================

print("\n" + "=" * 70)
print("AFTER TRAINING - Final Evaluation")
print("=" * 70)

tinystories_ppl_after = evaluate_tinystories(model, tokenizer, args.device)
wikitext_ppl_after = evaluate_wikitext103(model, tokenizer, args.device)

# ============================================================
# 性能对比
# ============================================================

print("\n" + "=" * 70)
print("Performance Comparison")
print("=" * 70)
print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
print("-" * 70)

# TinyStories
tinystories_change = tinystories_ppl_after - tinystories_ppl_before
tinystories_improve = ((tinystories_ppl_before - tinystories_ppl_after) / tinystories_ppl_before) * 100
print(f"{'TinyStories PPL':<30} {tinystories_ppl_before:<15.2f} {tinystories_ppl_after:<15.2f} {tinystories_change:+.2f} ({tinystories_improve:+.1f}%)")

# WikiText-103
wikitext_change = wikitext_ppl_after - wikitext_ppl_before
forgetting_pct = (wikitext_change / wikitext_ppl_before) * 100
print(f"{'WikiText-103 PPL':<30} {wikitext_ppl_before:<15.2f} {wikitext_ppl_after:<15.2f} {wikitext_change:+.2f}")
print(f"{'Forgetting':<30} {'':<15} {'':<15} {forgetting_pct:+.2f}%")

print("=" * 70)

# 保存结果
results = {
    'experiment_name': run_name,
    'freeze_strategy': freeze_strategy_names.get(args.weight_frozen, 'custom'),
    'num_trainable': num_trainable,
    'num_frozen': num_frozen,
    'trainable_ratio': num_trainable / (num_trainable + num_frozen),
    'before_training': {
        'tinystories_ppl': tinystories_ppl_before,
        'wikitext103_ppl': wikitext_ppl_before,
    },
    'after_training': {
        'tinystories_ppl': tinystories_ppl_after,
        'wikitext103_ppl': wikitext_ppl_after,
    },
    'performance_change': {
        'tinystories_ppl_change': tinystories_change,
        'tinystories_improvement_pct': tinystories_improve,
        'wikitext103_ppl_change': wikitext_change,
        'forgetting_percentage': forgetting_pct,
    },
    'config': vars(args)
}

with open(f"{output_dir}/continual_learning_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_dir}/continual_learning_results.json")

wandb.log({
    'final/tinystories_ppl': tinystories_ppl_after,
    'final/wikitext103_ppl': wikitext_ppl_after,
    'final/tinystories_improvement_pct': tinystories_improve,
    'final/forgetting_pct': forgetting_pct,
})

wandb.finish()

print("\n" + "=" * 70)
print("✓ Training completed!")
print("=" * 70)
