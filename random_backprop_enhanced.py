#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Backpropagation Experiment - Enhanced Version
支持三种场景：
1. 冻结参数场景下的random backprop
2. 全参数训练场景下的random backprop
3. 训练中途关闭random backprop
"""
 
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

class RandomBackpropManager:
    """管理随机反向传播的核心类 - 增强版"""

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "none",
        projection_type: str = "random",
        projection_rank: Optional[int] = None,
        resample_every_batch: bool = False,
        random_seed: Optional[int] = None,
        device: str = "cuda",
        allow_trainable: bool = False,  # 新增：允许对可训练参数应用random backprop
    ):
        """
        Args:
            allow_trainable: 如果为True，允许对可训练参数应用random backprop
                           （用于全参数训练场景）
        """
        self.model = model
        self.strategy = strategy
        self.original_strategy = strategy  # 保存原始策略，用于恢复
        self.projection_type = projection_type
        self.projection_rank = projection_rank
        self.resample_every_batch = resample_every_batch
        self.random_seed = random_seed
        self.device = device
        self.allow_trainable = allow_trainable

        # 存储原始参数和对应的随机替代
        self.frozen_params: Dict[str, nn.Parameter] = {}
        self.random_replacements: Dict[str, torch.Tensor] = {}
        self.backward_hooks: List = []
        self.wrapped_modules: Dict[str, nn.Module] = {}  # 存储被包装的模块

        # 随机数生成器
        self.rng = torch.Generator(device=device)
        if random_seed is not None:
            self.rng.manual_seed(random_seed)

        # 状态标志
        self.is_enabled = True

        print("=" * 70)
        print(f"Random Backpropagation Manager Initialized (Enhanced)")
        print(f"  Strategy: {strategy}")
        print(f"  Allow Trainable: {allow_trainable}")
        if strategy == "low_rank_projection":
            print(f"  Projection Type: {projection_type}")
            print(f"  Projection Rank: {projection_rank if projection_rank else 'Full Rank'}")
        print(f"  Resample Every Batch: {resample_every_batch}")
        print(f"  Random Seed: {random_seed if random_seed else 'Dynamic'}")
        print("=" * 70)

    def register_frozen_layer(self, name: str, param: nn.Parameter):
        """注册一个参数（可以是冻结的或可训练的）"""
        self.frozen_params[name] = param
        trainable_status = "trainable" if param.requires_grad else "frozen"
        print(f"  Registered {trainable_status} param: {name:60s} shape={tuple(param.shape)}")

    def _generate_random_matrix(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机矩阵"""
        if self.random_seed is not None and not self.resample_every_batch:
            return torch.randn(shape, device=self.device, generator=self.rng) * 0.02
        else:
            return torch.randn(shape, device=self.device) * 0.02

    def _generate_rotation_matrix(self, n: int) -> torch.Tensor:
        """生成n×n的正交旋转矩阵（使用QR分解）"""
        if self.random_seed is not None and not self.resample_every_batch:
            A = torch.randn(n, n, device=self.device, generator=self.rng)
        else:
            A = torch.randn(n, n, device=self.device)

        Q, R = torch.linalg.qr(A)
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        return Q

    def _generate_projection_matrix(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """生成投影矩阵"""
        if self.projection_type == "rotation":
            if in_dim == out_dim:
                return self._generate_rotation_matrix(in_dim)
            else:
                max_dim = max(in_dim, out_dim)
                Q = self._generate_rotation_matrix(max_dim)
                return Q[:out_dim, :in_dim]
        else:
            proj = self._generate_random_matrix((out_dim, in_dim))
            proj = proj / np.sqrt(in_dim)
            return proj

    def initialize_random_replacements(self):
        """初始化所有参数的随机替代"""
        print("=" * 70)
        print("Initializing Random Replacements for Backpropagation")
        print("=" * 70)

        for name, param in self.frozen_params.items():
            if self.strategy == "full_random":
                random_param = self._generate_random_matrix(param.shape)
                self.random_replacements[name] = random_param
                print(f"  [Full Random] {name:50s} shape={tuple(param.shape)}")

            elif self.strategy == "low_rank_projection":
                if "weight" in name and len(param.shape) == 2:
                    out_dim, in_dim = param.shape
                    if self.projection_rank is not None:
                        proj_dim = min(self.projection_rank, in_dim, out_dim)
                    else:
                        proj_dim = in_dim

                    proj_matrix = self._generate_projection_matrix(in_dim, proj_dim)
                    self.random_replacements[name] = proj_matrix
                    print(f"  [Projection] {name:50s} original={tuple(param.shape)} -> proj={(proj_dim, in_dim)}")
                else:
                    random_param = self._generate_random_matrix(param.shape)
                    self.random_replacements[name] = random_param
                    print(f"  [Random] {name:50s} shape={tuple(param.shape)}")

        print("=" * 70)
        print(f"Total {len(self.random_replacements)} random replacements initialized")
        print("=" * 70 + "\n")

    def disable(self):
        """关闭random backprop，恢复标准训练"""
        print("\n" + "=" * 70)
        print("DISABLING RANDOM BACKPROPAGATION - Switching to Standard Training")
        print("=" * 70)

        self.is_enabled = False
        self.strategy = "none"

        # 移除所有hooks
        self.remove_hooks()

        # 恢复所有被包装模块的原始forward方法
        for name, module in self.wrapped_modules.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
                print(f"  Restored original forward for: {name}")

        print("=" * 70)
        print("Random backpropagation disabled. Now using standard backpropagation.")
        print("=" * 70 + "\n")

    def enable(self):
        """重新启用random backprop"""
        if self.original_strategy == "none":
            print("Cannot enable: original strategy was 'none'")
            return

        print("\n" + "=" * 70)
        print("RE-ENABLING RANDOM BACKPROPAGATION")
        print("=" * 70)

        self.is_enabled = True
        self.strategy = self.original_strategy

        # 重新应用hooks和包装
        # 注意：需要重新调用wrap函数，这里简化处理
        print("Random backpropagation re-enabled")
        print("=" * 70 + "\n")

    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.backward_hooks:
            handle.remove()
        self.backward_hooks.clear()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "strategy": self.strategy,
            "is_enabled": self.is_enabled,
            "num_frozen_params": len(self.frozen_params),
            "num_hooks": len(self.backward_hooks),
            "resample_every_batch": self.resample_every_batch,
            "allow_trainable": self.allow_trainable,
        }

        if self.strategy == "low_rank_projection":
            stats["projection_type"] = self.projection_type
            stats["projection_rank"] = self.projection_rank

        return stats


class RandomBackwardLinear(torch.autograd.Function):
    """
    自定义autograd函数，用于实现随机反向传播
    Forward使用真实权重，Backward可选使用随机权重
    """

    @staticmethod
    def forward(ctx, input, weight, bias, random_weight, use_random_backward, manager):
        """
        Args:
            input: (batch, in_features)
            weight: (out_features, in_features)
            bias: (out_features,)
            random_weight: 用于backward的随机权重
            use_random_backward: 是否使用随机权重
            manager: RandomBackpropManager实例，用于检查状态
        """
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, random_weight)
        ctx.use_random_backward = use_random_backward
        ctx.manager = manager

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, random_weight = ctx.saved_tensors
        use_random_backward = ctx.use_random_backward
        manager = ctx.manager

        grad_input = grad_weight = grad_bias = None

        # 检查manager是否仍然启用
        if manager and not manager.is_enabled:
            use_random_backward = False

        # 选择用于梯度传播的权重
        backward_weight = random_weight if (use_random_backward and random_weight is not None) else weight

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(backward_weight)

        if ctx.needs_input_grad[1]:
            # 如果weight是可训练的，计算其梯度
            grad_weight = grad_output.t().matmul(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None


def wrap_frozen_layer_with_random_backward(
    module: nn.Module,
    random_manager: RandomBackpropManager,
    layer_name: str
):
    """
    包装一个层，使其在backward时使用随机参数
    """
    if not isinstance(module, nn.Linear):
        print("="*70)
        print("Did not pass the module")
        return
    print("="*70)
    print("Did pass the module")
    # 保存原始forward方法
    if not hasattr(module, '_original_forward'):
        module._original_forward = module.forward
        random_manager.wrapped_modules[layer_name] = module

    def new_forward(input):
        # 检查manager状态
        if not random_manager.is_enabled or random_manager.strategy == "none":
            return module._original_forward(input)

        
        # 获取随机替换参数
        weight_name = f"{layer_name}.weight"
        random_weight = random_manager.random_replacements.get(weight_name, None)

        # 如果需要每batch重新采样
        if random_manager.resample_every_batch and random_weight is not None:
            if random_manager.strategy == "full_random":
                random_weight = random_manager._generate_random_matrix(module.weight.shape)
            elif random_manager.strategy == "low_rank_projection" and len(module.weight.shape) == 2:
                out_dim, in_dim = module.weight.shape
                proj_dim = random_manager.projection_rank if random_manager.projection_rank else in_dim
                random_weight = random_manager._generate_projection_matrix(in_dim, min(proj_dim, in_dim, out_dim))
            random_manager.random_replacements[weight_name] = random_weight

        if random_weight is None:
            return module._original_forward(input)

        # 使用自定义autograd函数
        print("="*70)
        print("Apply autograd function to modules")
        use_random = random_manager.strategy in ["full_random", "low_rank_projection"]
        output = RandomBackwardLinear.apply(
            input,
            module.weight,
            module.bias,
            random_weight,
            use_random,
            random_manager
        )

        return output

    module.forward = new_forward


def setup_random_backprop_experiment(
    model,
    frozen_layer_names: List[str],
    strategy: str = "none",
    projection_type: str = "random",
    projection_rank: Optional[int] = None,
    resample_every_batch: bool = False,
    random_seed: Optional[int] = None,
    device: str = "cuda",
    allow_trainable: bool = False,  # 新增参数
) -> RandomBackpropManager:
    """
    设置随机反向传播实验

    Args:
        allow_trainable: 允许对可训练参数应用random backprop（用于全参数训练）
    """
    manager = RandomBackpropManager(
        model=model,
        strategy=strategy,
        projection_type=projection_type,
        projection_rank=projection_rank,
        resample_every_batch=resample_every_batch,
        random_seed=random_seed,
        device=device,
        allow_trainable=allow_trainable,
    )
    print("="*70)
    print("Did pass the module11111")
    # 注册参数
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in frozen_layer_names):
            # 如果allow_trainable=True，不检查requires_grad
            if allow_trainable or not param.requires_grad:
                manager.register_frozen_layer(name, param)

    # 初始化随机替换
    if strategy != "none":
        manager.initialize_random_replacements()
    print("="*70)
    print("Did pass the module22222")
    # 包装线性层
    if strategy in ["full_random", "low_rank_projection"]:
        print("="*70)
        print("Did pass the module?")
        for name, module in model.named_modules():
            print("="*70)
            print("Did pass the module??")
            print(name)
            print(module)
            if any(layer_name in name for layer_name in frozen_layer_names):
                print("="*70)
                print("Did pass the module?????")
                if isinstance(module, nn.Linear):
                    wrap_frozen_layer_with_random_backward(module, manager, name)
                    print(f"  Wrapped layer: {name}")

    return manager
