#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import accuracy, AverageMeter, ModelEma
import random
import numpy as np
import wandb
import os
from tqdm.auto import tqdm
import argparse
import json
import logging
from pathlib import Path

# 创建logger
logger = logging.getLogger(__name__)

# ============ 参数解析 ============
parser = argparse.ArgumentParser(description='TinyViT Experiments with Fine-grained Freezing')

# Model arguments
parser.add_argument('--model', type=str, default='tiny_vit_21m_224.in1k',
                    help='Model architecture from timm')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Use pretrained weights')
parser.add_argument('--num_classes', type=int, default=100,
                    help='Number of classes for ImageNet')
parser.add_argument('--img_size', type=int, default=224,
                    help='Input image size')

# Dataset arguments
parser.add_argument('--data_dir', type=str, default='/path/to/imagenet',
                    help='Path to ImageNet dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'torch/imagenet'])
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--val_split', type=str, default='validation')

# Training arguments
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size per GPU')
parser.add_argument('--workers', type=int, default=8,
                    help='Number of data loading workers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

# Optimizer arguments
parser.add_argument('--opt', type=str, default='adamw',
                    help='Optimizer (adamw, sgd, lamb)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD')

# Scheduler arguments
parser.add_argument('--sched', type=str, default='cosine',
                    help='LR scheduler (cosine, step, plateau)')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='Warmup epochs')
parser.add_argument('--warmup_lr', type=float, default=1e-6,
                    help='Warmup learning rate')
parser.add_argument('--min_lr', type=float, default=1e-5,
                    help='Minimum learning rate')

# Augmentation arguments
parser.add_argument('--color_jitter', type=float, default=0.4,
                    help='Color jitter factor')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',
                    help='AutoAugment policy')
parser.add_argument('--reprob', type=float, default=0.25,
                    help='Random erasing probability')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erasing mode')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erasing count')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='Mixup alpha')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='Cutmix alpha')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of applying mixup/cutmix')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing')

# Regularization arguments
parser.add_argument('--drop', type=float, default=0.0,
                    help='Dropout rate')
parser.add_argument('--drop_path', type=float, default=0.1,
                    help='Stochastic depth rate')

# Freezing and initialization arguments
parser.add_argument('--weight_frozen', type=int, default=0,
                    help='-1: zero-shot; '
                         '0: full finetune; '
                         '1: freeze attn+mlp in all TinyVitBlocks; '
                         '2: freeze attn only in all TinyVitBlocks; '
                         '3: freeze mlp only in all TinyVitBlocks; '
                         '4: freeze attn+mlp+patch_embed')
parser.add_argument('--freeze_stages', type=str, default='',
                    help='Specific stages to freeze (e.g., "0,1" or "1-2")')
parser.add_argument('--freeze_blocks', type=str, default='',
                    help='Specific blocks to freeze (e.g., "stage1:0-1,stage2:0-3")')
parser.add_argument('--from_scratch', action='store_true',
                    help='Initialize model from scratch')

# QKV initialization (adapted from LLM experiments)
parser.add_argument('--qkv_identity_init', type=str, default='none',
                    choices=['none', 'all', 'shallow', 'deep', 'mixed'],
                    help='QKV identity initialization strategy')
parser.add_argument('--identity_alpha', type=float, default=0.8,
                    help='Weight for identity component')
parser.add_argument('--identity_beta', type=float, default=0.2,
                    help='Weight for random component')
parser.add_argument('--freeze_qkv_components', type=str, default='all',
                    choices=['all', 'none', 'q', 'k', 'v', 'qk', 'qv', 'kv'],
                    help='Which QKV components to freeze')

# Logging arguments
parser.add_argument('--project_name', type=str, default='tinyvit_experiments',
                    help='W&B project name')
parser.add_argument('--run_name', type=str, default=None,
                    help='W&B run name')
parser.add_argument('--output_dir', type=str, default='./outputs_tinyvit',
                    help='Output directory')
parser.add_argument('--log_interval', type=int, default=50,
                    help='Logging interval')
parser.add_argument('--eval_interval', type=int, default=1,
                    help='Evaluation interval (epochs)')
parser.add_argument('--print_model', action='store_true',
                    help='Print model structure')

# EMA
parser.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable Model EMA')
parser.add_argument('--model_ema_decay', type=float, default=0.9999,
                    help='EMA decay rate')

args = parser.parse_args()

# ============ 设置随机种子 ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# ============ 冻结策略名称 ============
freeze_strategy_names = {
    -1: "zeroshot",
    0: "full_finetune",
    1: "freeze_attn_mlp",
    2: "freeze_attn_only",
    3: "freeze_mlp_only",
    4: "freeze_attn_mlp_patch"
}

# ============ 初始化 wandb ============
run_name = args.run_name or f"{freeze_strategy_names.get(args.weight_frozen, 'custom')}_{args.model}_seed{args.seed}"
if args.qkv_identity_init != 'none':
    run_name += f"_qkv_{args.qkv_identity_init}"

wandb.init(project=args.project_name, name=run_name, config=vars(args))

# ============ QKV组件管理工具函数 ============
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

# ============ 模型初始化函数 ============
def create_model(args):
    """创建或加载Vision Transformer模型（通用版本）"""
    print(f"\n{'='*70}")
    print(f"Creating model: {args.model}")
    print(f"{'='*70}")
    
    # 从模型名称中提取输入尺寸（如果有）
    # 例如：tiny_vit_21m_224 -> 224
    import re
    size_match = re.search(r'_(\d+)(?:\.in1k)?$', args.model)
    if size_match:
        model_size = int(size_match.group(1))
        if args.img_size != model_size:
            print(f"⚠ Model {args.model} expects {model_size}x{model_size} input")
            print(f"  Updating args.img_size from {args.img_size} to {model_size}")
            args.img_size = model_size
    
    # 检查模型是否支持特定参数
    model_cfg = timm.get_pretrained_cfg(args.model) if args.pretrained else None
    
    # 构建模型参数
    model_kwargs = {
        'num_classes': args.num_classes,
    }
    
    # 尝试添加可选参数（如果模型支持）
    try:
        # 先创建一个临时模型来检查支持的参数
        import inspect
        model_fn = timm.models.registry.model_entrypoint(args.model)
        sig = inspect.signature(model_fn)
        supported_params = set(sig.parameters.keys())
        
        # 根据支持的参数添加配置
        if 'drop_rate' in supported_params and args.drop > 0:
            model_kwargs['drop_rate'] = args.drop
        
        if 'drop_path_rate' in supported_params and args.drop_path > 0:
            model_kwargs['drop_path_rate'] = args.drop_path
        
        if 'img_size' in supported_params:
            model_kwargs['img_size'] = args.img_size
        
        print(f"Model supports parameters: {supported_params}")
        print(f"Using parameters: {model_kwargs}")
        
    except Exception as e:
        print(f"Could not inspect model parameters: {e}")
        print("Using minimal parameter set")
    
    # 创建模型
    if args.from_scratch:
        print("Initializing from scratch...")
        model = timm.create_model(
            args.model,
            pretrained=False,
            **model_kwargs
        ) 
    else:
        print(f"Loading pretrained weights...")
        model = timm.create_model(
            args.model,
            pretrained=True,
            **model_kwargs
        )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Input size: {args.img_size}x{args.img_size}")
    print(f"{'='*70}\n")
    
    return model


# ============ 打印模型结构 ============
def print_model_structure(model, verbose=True):
    """打印TinyViT模型结构"""
    
    print("\n" + "="*70)
    print("TINYVIT MODEL STRUCTURE")
    print("="*70)
    
    # 1. 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")
    
    # 2. 主要组件
    print("\n[Main Components]")
    if hasattr(model, 'patch_embed'):
        print(f"  • Patch Embedding: {model.patch_embed.__class__.__name__}")
    
    if hasattr(model, 'stages'):
        print(f"  • Stages: {len(model.stages)} stages")
        for stage_idx, stage in enumerate(model.stages):
            stage_name = stage.__class__.__name__
            if hasattr(stage, 'blocks'):
                num_blocks = len(stage.blocks)
                print(f"    Stage {stage_idx} ({stage_name}): {num_blocks} blocks")
                
                # 打印第一个TinyVitBlock的结构
                for block_idx, block in enumerate(stage.blocks):
                    if block.__class__.__name__ == 'TinyVitBlock':
                        print(f"      Block {block_idx} (TinyVitBlock):")
                        if hasattr(block, 'attn'):
                            attn = block.attn
                            if hasattr(attn, 'qkv'):
                                print(f"        - Attention QKV: {attn.qkv.weight.shape}")
                        if hasattr(block, 'mlp'):
                            print(f"        - MLP: {block.mlp.__class__.__name__}")
                        break  # 只打印第一个block
    
    if hasattr(model, 'head'):
        print(f"  • Classification Head: {model.head.__class__.__name__}")
    
    if verbose:
        # 3. 所有命名参数
        print("\n[All Named Parameters]")
        for name, param in model.named_parameters():
            trainable = "✓" if param.requires_grad else "✗"
            print(f"  {trainable} {name:70s} {tuple(param.shape)}")
    
    print("="*70 + "\n")

# ============ QKV特殊初始化函数 ============
def initialize_qkv_with_identity(model, strategy='all', alpha=0.8, beta=0.2):
    """
    对TinyViT的attention层QKV应用单位矩阵初始化
    """
    if strategy == 'none':
        print("\n[QKV Init] Strategy is 'none', keeping default initialization")
        return 0
    
    print("\n" + "="*70)
    print(f"Applying QKV initialization (strategy: {strategy})")
    if strategy == 'mixed':
        print(f" Alpha (identity weight): {alpha}")
        print(f" Beta (random weight): {beta}")
    print("="*70)
    
    initialized_count = 0
    total_blocks = 0
    
    # 遍历所有stages
    if hasattr(model, 'stages'):
        for stage_idx, stage in enumerate(model.stages):
            if hasattr(stage, 'blocks'):
                for block_idx, block in enumerate(stage.blocks):
                    total_blocks += 1
                    
                    # 只处理TinyVitBlock
                    if block.__class__.__name__ != 'TinyVitBlock':
                        continue
                    
                    # 确定是否对当前block应用初始化
                    should_initialize = False
                    if strategy == 'all':
                        should_initialize = True
                    elif strategy == 'shallow':
                        # 只初始化前两个stage的blocks
                        should_initialize = (stage_idx < 2)
                    elif strategy == 'deep':
                        # 只初始化后两个stage的blocks
                        num_stages = len(model.stages)
                        should_initialize = (stage_idx >= num_stages - 2)
                    elif strategy == 'mixed':
                        should_initialize = True
                    
                    if not should_initialize:
                        continue
                    
                    # 查找attention层
                    if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                        attn = block.attn
                        qkv_weight = attn.qkv.weight
                        qkv_bias = attn.qkv.bias
                        
                        # QKV weight shape: [3*embed_dim, embed_dim]
                        embed_dim = qkv_weight.shape[1]
                        
                        with torch.no_grad():
                            for i, component in enumerate(['q', 'k', 'v']):
                                start_idx = i * embed_dim
                                end_idx = (i + 1) * embed_dim
                                
                                if strategy in ['identity', 'all', 'shallow', 'deep']:
                                    # 完全单位矩阵
                                    identity_matrix = torch.eye(
                                        embed_dim,
                                        device=qkv_weight.device,
                                        dtype=qkv_weight.dtype
                                    )
                                    qkv_weight[start_idx:end_idx, :] = identity_matrix
                                
                                elif strategy == 'mixed':
                                    # 混合初始化
                                    current_random = qkv_weight[start_idx:end_idx, :].clone()
                                    identity_part = alpha * torch.eye(
                                        embed_dim,
                                        device=qkv_weight.device,
                                        dtype=qkv_weight.dtype
                                    )
                                    random_part = beta * current_random
                                    qkv_weight[start_idx:end_idx, :] = identity_part + random_part
                            
                            # bias初始化为0
                            if qkv_bias is not None:
                                qkv_bias.zero_()
                        
                        print(f" Stage {stage_idx}, Block {block_idx}: QKV initialized ({strategy})")
                        initialized_count += 1
    
    print("="*70)
    print(f"Applied initialization to {initialized_count}/{total_blocks} blocks")
    print("="*70 + "\n")
    
    return initialized_count

# ============ 冻结策略函数 ============
def apply_freeze_strategy(model, strategy, freeze_stages_str='', freeze_blocks_str='', freeze_qkv_components='all'):
    """
    应用冻结策略到TinyViT
    
    策略说明:
    -1: Zero-shot (冻结所有参数)
     0: Full finetune (不冻结)
     1: Freeze attn + mlp in all TinyVitBlocks
     2: Freeze attn only in all TinyVitBlocks
     3: Freeze mlp only in all TinyVitBlocks
     4: Freeze attn + mlp + patch_embed
    """
    print("\n" + "="*70)
    print(f"Applying freeze strategy: {freeze_strategy_names.get(strategy, 'custom')}")
    print(f"QKV component control: {freeze_qkv_components}")
    if freeze_stages_str:
        print(f"Custom freeze stages: {freeze_stages_str}")
    if freeze_blocks_str:
        print(f"Custom freeze blocks: {freeze_blocks_str}")
    print("="*70)
    
    num_frozen = 0
    num_trainable = 0
    qkv_components = parse_qkv_components(freeze_qkv_components)
    
    # 解析自定义冻结配置
    freeze_stage_indices = set()
    if freeze_stages_str:
        for part in freeze_stages_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                freeze_stage_indices.update(range(start, end + 1))
            else:
                freeze_stage_indices.add(int(part))
    
    # 解析block级别的冻结配置
    # 格式: "stage0:0-1,stage1:0-3" 或 "0:0-1,1:0-3"
    freeze_block_map = {}  # {stage_idx: [block_indices]}
    if freeze_blocks_str:
        for stage_spec in freeze_blocks_str.split(','):
            stage_spec = stage_spec.strip()
            if ':' in stage_spec:
                stage_part, blocks_part = stage_spec.split(':')
                # 提取stage索引
                stage_idx = int(stage_part.replace('stage', ''))
                
                # 解析block索引
                block_indices = []
                for block_range in blocks_part.split(';'):
                    if '-' in block_range:
                        start, end = map(int, block_range.split('-'))
                        block_indices.extend(range(start, end + 1))
                    else:
                        block_indices.append(int(block_range))
                
                freeze_block_map[stage_idx] = block_indices
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        # 确定当前参数属于哪个stage和block
        current_stage_idx = None
        current_block_idx = None
        
        if 'stages.' in name:
            # 提取stage索引
            stage_str = name.split('stages.')[1].split('.')[0]
            if stage_str.isdigit():
                current_stage_idx = int(stage_str)
            
            # 提取block索引
            if '.blocks.' in name:
                block_str = name.split('.blocks.')[1].split('.')[0]
                if block_str.isdigit():
                    current_block_idx = int(block_str)
        
        # 基础冻结策略
        if strategy == -1:  # Zero-shot: freeze everything
            should_freeze = True
            
        elif strategy == 0:  # Full finetune: freeze nothing
            should_freeze = False
            
        elif strategy == 1:  # Freeze attn + mlp in all TinyVitBlocks
            if '.attn.' in name or '.mlp.' in name:
                # 检查是否在TinyVitBlock中
                if '.blocks.' in name:
                    should_freeze = True
                    
        elif strategy == 2:  # Freeze attn only in all TinyVitBlocks
            if '.attn.' in name and '.blocks.' in name:
                should_freeze = True
                
        elif strategy == 3:  # Freeze mlp only in all TinyVitBlocks
            if '.mlp.' in name and '.blocks.' in name:
                should_freeze = True
                
        elif strategy == 4:  # Freeze attn + mlp + patch_embed
            if any(x in name for x in ['.attn.', '.mlp.', 'patch_embed']):
                should_freeze = True
        
        # 自定义stage级别冻结
        if freeze_stage_indices and current_stage_idx is not None:
            if current_stage_idx in freeze_stage_indices:
                should_freeze = True
        
        # 自定义block级别冻结
        if freeze_block_map and current_stage_idx is not None and current_block_idx is not None:
            if current_stage_idx in freeze_block_map:
                if current_block_idx in freeze_block_map[current_stage_idx]:
                    should_freeze = True
        
        # 特殊处理 QKV 组件（如果只冻结部分 QKV）
        if '.qkv.' in name and len(qkv_components) < 3 and should_freeze:
            # 部分冻结 QKV 需要梯度，但会通过 hook 来 mask
            param.requires_grad = True
            num_trainable += param.numel()
            print(f" [PARTIAL-QKV] {name:70s} (freeze: {qkv_components})")
        else:
            param.requires_grad = not should_freeze
            if should_freeze:
                num_frozen += param.numel()
                # 详细标注冻结类型
                status = "FROZEN"
                if '.attn.' in name:
                    status = "FROZEN-ATTN"
                elif '.mlp.' in name:
                    status = "FROZEN-MLP"
                elif 'patch_embed' in name:
                    status = "FROZEN-PATCH"
                
                stage_block_info = ""
                if current_stage_idx is not None:
                    stage_block_info = f"[S{current_stage_idx}"
                    if current_block_idx is not None:
                        stage_block_info += f":B{current_block_idx}"
                    stage_block_info += "]"
                
                print(f" [{status:15s}] {stage_block_info:10s} {name:70s} {tuple(param.shape)}")
            else:
                num_trainable += param.numel()
                print(f" [TRAINABLE] {name:70s} {tuple(param.shape)}")
    
    total_params = num_trainable + num_frozen
    print("="*70)
    print(f"Trainable: {num_trainable:,} ({num_trainable/total_params*100:.2f}%)")
    print(f"Frozen: {num_frozen:,} ({num_frozen/total_params*100:.2f}%)")
    print(f"Total: {total_params:,}")
    print("="*70 + "\n")
    
    wandb.log({
        'num_trainable': num_trainable,
        'num_frozen': num_frozen,
        'trainable_ratio': num_trainable / total_params,
        'freeze_strategy': strategy
    })
    
    return num_trainable, num_frozen

# ============ 打印冻结统计 ============
def print_freeze_statistics(model):
    """打印详细的冻结统计信息"""
    print("\n" + "="*70)
    print("FREEZE STATISTICS BY COMPONENT")
    print("="*70)
    
    stats = {
        'patch_embed': {'trainable': 0, 'frozen': 0},
        'stages.downsample': {'trainable': 0, 'frozen': 0},
        'stages.blocks.attn': {'trainable': 0, 'frozen': 0},
        'stages.blocks.mlp': {'trainable': 0, 'frozen': 0},
        'stages.blocks.local_conv': {'trainable': 0, 'frozen': 0},
        'stages.blocks.other': {'trainable': 0, 'frozen': 0},
        'stages.MBConv': {'trainable': 0, 'frozen': 0},
        'head': {'trainable': 0, 'frozen': 0},
        'other': {'trainable': 0, 'frozen': 0}
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        
        # 分类参数
        if 'patch_embed' in name:
            component = 'patch_embed'
        elif 'downsample' in name:
            component = 'stages.downsample'
        elif '.blocks.' in name:
            if '.attn.' in name:
                component = 'stages.blocks.attn'
            elif '.mlp.' in name:
                component = 'stages.blocks.mlp'
            elif '.local_conv' in name:
                component = 'stages.blocks.local_conv'
            else:
                component = 'stages.blocks.other'
        elif 'MBConv' in name or 'conv' in name:
            component = 'stages.MBConv'
        elif 'head' in name:
            component = 'head'
        else:
            component = 'other'
        
        if is_trainable:
            stats[component]['trainable'] += num_params
        else:
            stats[component]['frozen'] += num_params
    
    # 打印统计
    print(f"\n{'Component':<30} {'Trainable':<20} {'Frozen':<20} {'Total':<20} {'Status':<15}")
    print("-" * 110)
    
    for component, counts in stats.items():
        trainable = counts['trainable']
        frozen = counts['frozen']
        total = trainable + frozen
        
        if total == 0:
            continue
        
        if frozen == 0:
            status = "✓ All Trainable"
        elif trainable == 0:
            status = "✗ All Frozen"
        else:
            status = "⚠ Partial"
        
        print(f"{component:<30} {trainable:>12,} ({trainable/total*100:>5.1f}%)  "
              f"{frozen:>12,} ({frozen/total*100:>5.1f}%)  "
              f"{total:>12,}        {status:<15}")
    
    print("="*70 + "\n")

# ============ 创建优化器（支持QKV部分冻结） ============
def create_optimizer_with_qkv_control(model, qkv_components_to_freeze, args):
    """
    创建优化器，支持QKV组件的细粒度控制
    """
    if len(qkv_components_to_freeze) < 3 and len(qkv_components_to_freeze) > 0:
        print("\n" + "="*70)
        print("Creating optimizer with QKV component control...")
        print(f"Freezing QKV components: {qkv_components_to_freeze}")
        print("="*70)
        
        def create_qkv_mask_hook(qkv_components_frozen):
            """创建 hook 来 mask 掉 frozen QKV 组件的梯度"""
            def hook(grad):
                if grad is None:
                    return None
                
                # QKV 权重形状: [3*embed_dim, embed_dim]
                # 或者 [3*embed_dim] for bias
                total_dim = grad.shape[0]
                embed_dim = total_dim // 3
                
                masked_grad = grad.clone()
                
                for i, component in enumerate(['q', 'k', 'v']):
                    if component in qkv_components_frozen:
                        start_idx = i * embed_dim
                        end_idx = (i + 1) * embed_dim
                        
                        # 清零 frozen 组件的梯度
                        if len(grad.shape) == 2:  # weight
                            masked_grad[start_idx:end_idx, :] = 0
                        else:  # bias
                            masked_grad[start_idx:end_idx] = 0
                
                return masked_grad
            
            return hook

        # 注册 hooks
        hook_count = 0
        for name, param in model.named_parameters():
            if '.qkv.' in name and ('weight' in name or 'bias' in name):
                param.register_hook(create_qkv_mask_hook(qkv_components_to_freeze))
                print(f" Registered gradient mask hook for {name}")
                hook_count += 1
        
        print(f"✓ Registered {hook_count} gradient hooks")
        print("="*70 + "\n")
    
    # 创建优化器
    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    
    print(f"✓ Optimizer: {args.opt} (lr={args.lr}, wd={args.weight_decay})")
    
    return optimizer

# ============ 数据加载 ============
def create_dataloaders(args):
    """使用 timm/mini-imagenet 数据集"""
    import os
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    import torch
    
    print(f"\n{'='*70}")
    print("Loading timm/mini-imagenet from Hugging Face...")
    print(f"{'='*70}")
    
    # 设置缓存目录
    cache_dir = "/work/hm235/random_transformer/data/mini_imagenet_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 下载数据集
        print(f"Cache directory: {cache_dir}")
        print("Downloading... (first run may take 10-30 minutes for ~13 GB)")
        
        dataset = load_dataset(
            "timm/mini-imagenet",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print(f"✓ Download complete!")
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['validation'])}")
        
    except Exception as e:
        print(f"✗ Failed to load mini-imagenet: {e}")
        print("  Possible issues:")
        print("  1. Network connection")
        print("  2. Disk quota (needs ~13 GB)")
        raise
    
    
    # ImageNet 标准统计
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # 包装 Hugging Face Dataset 为 PyTorch Dataset
    class HFImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            
            # 获取图片和标签
            image = item['image']
            label = item['label']
            
            # 确保是 PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            # 转换为 RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # 创建 PyTorch Dataset
    train_dataset = HFImageNetDataset(dataset['train'], transform=train_transform)
    val_dataset = HFImageNetDataset(dataset['validation'], transform=val_transform)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False
    )
    
    data_config = {
        'input_size': (3, args.img_size, args.img_size),
        'mean': mean,
        'std': std
    }
    
    print(f"\n✓ mini-imagenet loaded successfully")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Num classes: {args.num_classes}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, data_config


# ============ Mixup设置 ============
def create_mixup(args):
    """创建Mixup/Cutmix增强"""
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=None,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode='batch',
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        print(f"✓ Mixup enabled (mixup={args.mixup}, cutmix={args.cutmix})")
    else:
        mixup_fn = None
        print("✗ Mixup disabled")
    
    return mixup_fn

# ============ 损失函数 ============
def create_loss(args, mixup_fn):
    """创建损失函数"""
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"✓ Loss function: {criterion.__class__.__name__}")
    return criterion

# ============ 训练函数 ============
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, args, 
                    scheduler=None, mixup_fn=None):
    """训练一个 epoch - 支持 timm 损失函数"""
    import torch
    import torch.nn.functional as F
    from timm.utils import AverageMeter, accuracy
    
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # 检测是否使用 timm 的软标签损失
    use_soft_target = (
    type(criterion).__name__ in ['LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy']
    or 'timm.loss' in str(type(criterion))
    )
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        
        # 保存原始整数标签用于计算准确率
        targets_int = targets.clone()
        
        # 如果使用 mixup/cutmix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
            # mixup 后 targets 已经是软标签
        elif use_soft_target:
            # 如果没有 mixup 但使用软标签损失，需要转换为 one-hot
            targets = F.one_hot(targets, num_classes=args.num_classes).float()
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=args.use_amp if hasattr(args, 'use_amp') else False):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        
        if hasattr(args, 'use_amp') and args.use_amp:
            # 使用混合精度训练
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        
        # 计算准确率（使用原始整数标签）
        if mixup_fn is None:
            acc1, acc5 = accuracy(outputs, targets_int, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        
        # 打印进度
        if batch_idx % 10 == 0:
            if mixup_fn is None:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {losses.avg:.4f} '
                      f'Acc@1: {top1.avg:.2f} '
                      f'Acc@5: {top5.avg:.2f} '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            else:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {losses.avg:.4f} '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    # 学习率调度
    if scheduler is not None and hasattr(args, 'sched') and args.sched == 'cosine':
        scheduler.step(epoch)

    return losses.avg, top1.avg if mixup_fn is None else 0.0, top5.avg if mixup_fn is None else 0.0


#============== 验证函数 ============
@torch.no_grad()
def validate(epoch, model, val_loader, criterion, args):
    """验证函数 - 支持 timm 损失函数"""
    import torch
    import torch.nn.functional as F
    from timm.utils import AverageMeter, accuracy
    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # 检测是否使用 timm 的软标签损失
    use_soft_target = (
    type(criterion).__name__ in ['LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy']
    or 'timm.loss' in str(type(criterion))
    )

    
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        
        # 保存原始整数标签用于计算准确率
        targets_int = targets
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=args.use_amp if hasattr(args, 'use_amp') else False):
            outputs = model(images)
            
            # 如果使用 timm 损失函数，需要转换标签为 one-hot
            if use_soft_target:
                targets_onehot = F.one_hot(targets, num_classes=args.num_classes).float()
                loss = criterion(outputs, targets_onehot)
            else:
                loss = criterion(outputs, targets)
        
        # 计算准确率（使用原始整数标签）
        acc1, acc5 = accuracy(outputs, targets_int, topk=(1, 5))
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        # 打印进度
        if batch_idx % 10 == 0:
            print(f'Val: [{batch_idx}/{len(val_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Acc@1: {top1.avg:.2f} '
                  f'Acc@5: {top5.avg:.2f}')
    
    print(f'\n * Validation: Loss {losses.avg:.4f} Acc@1 {top1.avg:.2f} Acc@5 {top5.avg:.2f}\n')
    
    return losses.avg, top1.avg, top5.avg


# ============ 主训练流程 ============
def main():
    # 创建输出目录
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 创建模型
    model = create_model(args)
    model = model.to(args.device)
    
    # 打印模型结构
    if args.print_model:
        print_model_structure(model, verbose=True)
    
    # 应用QKV特殊初始化
    if args.qkv_identity_init != 'none':
        initialize_qkv_with_identity(
            model,
            strategy=args.qkv_identity_init,
            alpha=args.identity_alpha,
            beta=args.identity_beta
        )
    


    # 应用冻结策略
    qkv_components_to_freeze = parse_qkv_components(args.freeze_qkv_components)
    num_trainable, num_frozen = apply_freeze_strategy(
        model,
        args.weight_frozen,
        freeze_stages_str=args.freeze_stages,
        freeze_blocks_str=args.freeze_blocks,
        freeze_qkv_components=args.freeze_qkv_components
    )
    
    # 打印冻结统计
    print_freeze_statistics(model)
    
    # 创建数据加载器
    train_loader, eval_loader, data_config = create_dataloaders(args)
    
    # 创建Mixup
    mixup_fn = create_mixup(args)
    
    # 创建损失函数
    criterion = create_loss(args, mixup_fn)

    if args.weight_frozen == -1:
        # 零样本模式：只评估，不训练
        print("\n" + "="*70)
        print("ZERO-SHOT MODE: Skipping training")
        print("="*70)
        print("BEFORE TRAINING - Zero-shot Performance")
        print("="*70)
        val_loss, val_top1, val_top5 = validate(0, model, eval_loader, criterion, args)
        # 保存初始结果
        results = {
            'before_training': {
                'val_loss': val_loss,
                'val_top1_acc': val_top1,
                'val_top5_acc': val_top5
            }
        }
        results['after_training'] = results['before_training'].copy()
        results['config'] = vars(args)
        results['num_trainable'] = num_trainable
        results['num_frozen'] = num_frozen
        results['mode'] = 'zero_shot'
        
        # 保存结果
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Zero-shot results saved to {output_dir / 'results.json'}")
        
        # 记录到wandb
        wandb.log({
            'zero_shot/val_loss': val_loss,
            'zero_shot/val_top1_acc': val_top1,
            'zero_shot/val_top5_acc': val_top5
        })
        
        wandb.finish()
        print("\n✓ Zero-shot evaluation completed!")
        return
    
    # 创建优化器（支持QKV控制）
    optimizer = create_optimizer_with_qkv_control(
        model, 
        qkv_components_to_freeze, 
        args
    )
    
    # 创建学习率调度器
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
        cooldown_epochs=0
    )
    print(f"✓ Scheduler: {args.sched}")
    
    # 创建EMA模型
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device=args.device
        )
        print(f"✓ Model EMA enabled (decay={args.model_ema_decay})")
    
    # 训练前评估
    print("\n" + "="*70)
    print("BEFORE TRAINING - Zero-shot Performance")
    print("="*70)
    val_loss, val_top1, val_top5 = validate(0, model, eval_loader, criterion, args)
    
    # 保存初始结果
    results = {
        'before_training': {
            'val_loss': val_loss,
            'val_top1_acc': val_top1,
            'val_top5_acc': val_top5
        }
    }
        
    # 训练循环
    if args.weight_frozen != -1:
        print("\n" + "="*70)
        print("TRAINING START")
        print("="*70 + "\n")
        
        best_top1 = 0.0
        
        for epoch in range(1, args.epochs + 1):
            # 训练一个epoch
            train_loss, train_top1, train_top5 = train_one_epoch(
                epoch, model, train_loader, criterion, optimizer, args,
                scheduler=scheduler, mixup_fn=mixup_fn
            )

            
            # 验证
            if epoch % args.eval_interval == 0:
                if model_ema is not None:
                    val_loss, val_top1, val_top5 = validate(
                        epoch, model_ema.module, eval_loader, criterion, args
                    )
                else:
                    val_loss, val_top1, val_top5 = validate(
                        epoch, model, eval_loader, criterion, args
                    )
                
                # 保存最佳模型
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_top1_acc': val_top1,
                        'args': vars(args)
                    }
                    if model_ema is not None:
                        checkpoint['model_ema_state_dict'] = model_ema.module.state_dict()
                    
                    torch.save(checkpoint, output_dir / 'best_model.pth')
                    print(f"✓ Saved best model (top1={val_top1:.2f}%)")
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step(epoch)
        
        # 加载最佳模型进行最终评估
        print("\n" + "="*70)
        print("AFTER TRAINING - Final Performance")
        print("="*70)
        
        checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        val_loss, val_top1, val_top5 = validate(args.epochs, model, eval_loader, criterion, args)
        
        results['after_training'] = {
            'val_loss': val_loss,
            'val_top1_acc': val_top1,
            'val_top5_acc': val_top5,
            'best_epoch': checkpoint['epoch']
        }
    
    # 保存最终结果
    results['config'] = vars(args)
    results['num_trainable'] = num_trainable
    results['num_frozen'] = num_frozen
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    
    # 打印性能对比
    print("\n" + "="*70)
    print("Performance Comparison")
    print("="*70)
    print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-"*70)
    
    before = results['before_training']
    after = results.get('after_training', before)
    
    for metric in ['val_loss', 'val_top1_acc', 'val_top5_acc']:
        before_val = before.get(metric, 0)
        after_val = after.get(metric, 0)
        
        if 'loss' in metric:
            change = after_val - before_val
            change_str = f"{change:+.4f}"
        else:
            change = after_val - before_val
            change_str = f"{change:+.2f}%"
        
        print(f"{metric:<30} {before_val:<15.4f} {after_val:<15.4f} {change_str:<15}")
    
    print("="*70)
    
    wandb.finish()
    print("\n✓ All done!")

if __name__ == '__main__':
    main()
