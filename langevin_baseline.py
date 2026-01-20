
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Langevin Dynamics Sampling Baseline for Random Backpropagation
在标准训练中直接对梯度添加噪声（SGLD风格）
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np


class LangevinNoiseManager:
    """管理Langevin动力学噪声注入"""

    def __init__(
        self,
        model: nn.Module,
        noise_scale: float = 0.01,
        apply_to_layers: str = 'all',  # 'all', 'embedding', 'attn', 'mlp'
        temperature: float = 1.0,
        use_preconditioner: bool = False,
        random_seed: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            noise_scale: 噪声强度（相当于SGLD中的学习率）
            apply_to_layers: 对哪些层应用噪声
            temperature: 温度参数（控制探索vs利用）
            use_preconditioner: 是否使用预条件（类似RMSprop）
            random_seed: 随机种子
            device: 设备
        """
        self.model = model
        self.noise_scale = noise_scale
        self.apply_to_layers = apply_to_layers
        self.temperature = temperature
        self.use_preconditioner = use_preconditioner
        self.device = device


        if random_seed is not None:
            self.rng = torch.Generator(device=torch.device(device))
            self.rng.manual_seed(random_seed)
        else:
            self.rng = None


        # 状态标志
        self.is_enabled = True

        # 预条件矩阵（如果使用）
        self.precond_dict: Dict[str, torch.Tensor] = {}
        if use_preconditioner:
            self._initialize_preconditioner()

        # 注册hooks
        self.hooks = []
        self._register_hooks()

        print("=" * 70)
        print("Langevin Dynamics Noise Manager Initialized")
        print(f"  Noise Scale: {noise_scale}")
        print(f"  Temperature: {temperature}")
        print(f"  Apply to: {apply_to_layers}")
        print(f"  Preconditioner: {use_preconditioner}")
        print(f"  Random Seed: {random_seed}")
        print("=" * 70)

    def _should_apply_noise(self, name: str) -> bool:
        """判断是否对该参数应用噪声"""
        if self.apply_to_layers == 'all':
            return True
        elif self.apply_to_layers == 'embedding':
            return 'wte' in name or 'wpe' in name or 'lm_head' in name
        elif self.apply_to_layers == 'attn':
            return 'attn' in name
        elif self.apply_to_layers == 'mlp':
            return 'mlp' in name
        else:
            return False

    def _initialize_preconditioner(self):
        """初始化预条件矩阵（类似RMSprop的累积平方梯度）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_apply_noise(name):
                # 初始化为小值，避免除0
                self.precond_dict[name] = torch.ones_like(param) * 1e-8

    def _update_preconditioner(self, name: str, grad: torch.Tensor, decay: float = 0.99):
        """更新预条件矩阵（EMA of squared gradients）"""
        if name in self.precond_dict:
            self.precond_dict[name] = (
                decay * self.precond_dict[name] + 
                (1 - decay) * grad.pow(2)
            )

    def _add_langevin_noise(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """
        添加Langevin噪声到梯度

        SGLD更新: θ_t+1 = θ_t - η∇L + √(2η·T)·ε
        等价于在梯度上: grad = ∇L - √(2η·T)/η·ε = ∇L - √(2T/η)·ε
        """
        if not self.is_enabled:
            return grad

        # 生成高斯噪声
        if self.rng is not None:
            # 使用固定的 generator (可复现)
            noise = torch.randn(
                grad.shape,
                dtype=grad.dtype,
                device=grad.device,
                generator=self.rng
            )
        else:
            # 使用默认随机数生成器
            noise = torch.randn_like(grad)

        # 应用预条件（如果启用）
        if self.use_preconditioner and name in self.precond_dict:
            # 更新预条件矩阵
            self._update_preconditioner(name, grad)

            # 预条件噪声: G^(-1/2) * noise
            precond = self.precond_dict[name]
            noise = noise / (torch.sqrt(precond) + 1e-8)

        # 计算噪声标准差: σ = √(2·temperature·noise_scale)
        noise_std = np.sqrt(2.0 * self.temperature * self.noise_scale)
        # noise_std = np.sqrt(2 * self.temperature * self.lr)
        # 添加噪声到梯度
        noisy_grad = grad + noise_std * noise

        return noisy_grad

    def _register_hooks(self):
        """注册backward hooks来注入噪声"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_apply_noise(name):
                # 注册hook
                hook = param.register_hook(
                    lambda grad, n=name: self._add_langevin_noise(grad, n)
                )
                self.hooks.append(hook)
                print(f"  Registered Langevin hook: {name:60s}")

    def disable(self):
        """禁用Langevin噪声"""
        print("\n" + "=" * 70)
        print("DISABLING LANGEVIN NOISE - Switching to Standard Training")
        print("=" * 70)
        self.is_enabled = False

    def enable(self):
        """重新启用Langevin噪声"""
        print("\n" + "=" * 70)
        print("RE-ENABLING LANGEVIN NOISE")
        print("=" * 70)
        self.is_enabled = True

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'is_enabled': self.is_enabled,
            'noise_scale': self.noise_scale,
            'temperature': self.temperature,
            'apply_to_layers': self.apply_to_layers,
            'use_preconditioner': self.use_preconditioner,
            'num_hooks': len(self.hooks)
        }
        return stats


# ============ 使用示例（集成到你的experiment_llm-2.py中）============

def setup_langevin_baseline(
    model,
    noise_scale: float = 0.01,
    apply_to_layers: str = 'embedding',  # 只对embedding加噪声
    temperature: float = 1.0,
    use_preconditioner: bool = False,
    random_seed: Optional[int] = None,
    device: str = 'cuda'
):
    """
    设置Langevin Dynamics Baseline

    使用示例:
        langevin_manager = setup_langevin_baseline(
            model=model,
            noise_scale=0.01,
            apply_to_layers='embedding',
            temperature=1.0
        )
    """
    manager = LangevinNoiseManager(
        model=model,
        noise_scale=noise_scale,
        apply_to_layers=apply_to_layers,
        temperature=temperature,
        use_preconditioner=use_preconditioner,
        random_seed=random_seed,
        device=device
    )
    return manager


# ============ Callback: 训练中途关闭Langevin噪声 ============

from transformers.trainer_callback import TrainerCallback
import wandb

class DisableLangevinCallback(TrainerCallback):
    """在训练特定步数后关闭Langevin噪声的回调"""

    def __init__(self, langevin_manager, disable_ratio, max_steps):
        """
        Args:
            langevin_manager: LangevinNoiseManager实例
            disable_ratio: 在总步数的这个比例后关闭 (0.9 = 90%)
            max_steps: 总训练步数
        """
        self.langevin_manager = langevin_manager
        self.disable_step = int(max_steps * disable_ratio)
        self.disabled = False

        print("=" * 70)
        print(f"DisableLangevinCallback Initialized")
        print(f"  Will disable at step: {self.disable_step} / {max_steps} ({disable_ratio*100:.0f}%)")
        print("=" * 70)

    def on_step_begin(self, args, state, control, **kwargs):
        """在每个训练步开始时检查是否需要关闭Langevin噪声"""
        if not self.disabled and state.global_step >= self.disable_step:
            print("\n" + "=" * 70)
            print(f"🔄 TRAINING MILESTONE: Step {state.global_step}/{args.max_steps}")
            print(f"   Disabling Langevin Noise")
            print(f"   Switching to Standard Gradient Descent")
            print("=" * 70 + "\n")

            self.langevin_manager.disable()
            self.disabled = True

            # 记录到wandb
            wandb.log({
                'langevin_noise_disabled': True,
                'disable_step': state.global_step,
                'training_phase': 'standard'
            })


# ============ 集成到训练流程 ============

"""
在你的experiment_llm-2.py中使用:

# 1. 冻结策略（与原来相同）
layers_to_freeze, _ = apply_freeze_strategy(model, args.weight_frozen, 'all')
num_frozen, num_trainable = execute_freeze(model, layers_to_freeze, ['q', 'k', 'v'])

# 2. 设置Langevin噪声（替代Random Backprop）
langevin_manager = setup_langevin_baseline(
    model=model,
    noise_scale=args.langevin_noise_scale,      # 新参数
    apply_to_layers='embedding',                # 只对embedding加噪声
    temperature=args.langevin_temperature,      # 新参数
    use_preconditioner=args.langevin_precond,   # 新参数
    random_seed=args.seed,
    device=args.device
)

# 3. 训练配置
training_args = TrainingArguments(...)

# 4. Callback设置
callbacks = [EvalCallback()]

if langevin_manager and args.disable_langevin_at_ratio < 1.0:
    disable_callback = DisableLangevinCallback(
        langevin_manager=langevin_manager,
        disable_ratio=args.disable_langevin_at_ratio,
        max_steps=args.max_steps
    )
    callbacks.append(disable_callback)

# 5. 训练
trainer = Trainer(model=model, args=training_args, callbacks=callbacks, ...)
trainer.train()
"""


# ============ 命令行参数（添加到argparse）============

"""
在experiment_llm-2.py的argparse部分添加:

parser.add_argument('--use_langevin_baseline', action='store_true',
                    help='Use Langevin dynamics noise instead of random backprop')
parser.add_argument('--langevin_noise_scale', type=float, default=0.01,
                    help='Noise scale for Langevin dynamics (η)')
parser.add_argument('--langevin_temperature', type=float, default=1.0,
                    help='Temperature parameter for Langevin sampling')
parser.add_argument('--langevin_precond', action='store_true',
                    help='Use preconditioner (RMSprop-style) for Langevin noise')
parser.add_argument('--langevin_apply_to', type=str, default='embedding',
                    choices=['all', 'embedding', 'attn', 'mlp'],
                    help='Which layers to apply Langevin noise')
parser.add_argument('--disable_langevin_at_ratio', type=float, default=1.0,
                    help='Disable Langevin noise after this ratio of training')
"""


# ============ 实验对比示例 ============

"""
实验1: Random Backpropagation (你的原方法)
python experiment_llm-2.py \
    --weight_frozen 1 \
    --random_backprop_strategy full_random \
    --seed 42 \
    --max_steps 10000

实验2: Langevin Dynamics Baseline
python experiment_llm-2.py \
    --weight_frozen 1 \
    --use_langevin_baseline \
    --langevin_noise_scale 0.01 \
    --langevin_temperature 1.0 \
    --langevin_apply_to embedding \
    --seed 42 \
    --max_steps 10000

实验3: Langevin + Preconditioner
python experiment_llm-2.py \
    --weight_frozen 1 \
    --use_langevin_baseline \
    --langevin_noise_scale 0.01 \
    --langevin_precond \
    --seed 42 \
    --max_steps 10000

实验4: 中途关闭噪声
python experiment_llm-2.py \
    --weight_frozen 1 \
    --use_langevin_baseline \
    --langevin_noise_scale 0.01 \
    --disable_langevin_at_ratio 0.9 \
    --seed 42 \
    --max_steps 10000
"""
