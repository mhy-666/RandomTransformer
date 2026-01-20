#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forward Propagation Experiment
与 Random Backprop 相反：Forward 使用随机权重，Backward 使用真实权重
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class RandomForwardManager:
    """管理随机前向传播的核心类"""
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "none",  # "none", "additive_noise", "full_random"
        noise_scale: float = 0.02,
        projection_type: str = "random",  # "random", "rotation"
        projection_rank: Optional[int] = None,
        resample_every_batch: bool = False,
        random_seed: Optional[int] = None,
        device: str = "cuda",
        allow_trainable: bool = False,
    ):
        """
        Args:
            strategy: 随机前向传播策略
                - "none": 正常训练
                - "additive_noise": W_forward = W_real + noise (推荐)
                - "full_random": W_forward = W_random (完全独立)
            noise_scale: 噪声幅度（用于 additive_noise）
            projection_type: 投影矩阵类型（用于 full_random）
            allow_trainable: 是否允许对可训练参数应用 random forward
        """
        self.model = model
        self.strategy = strategy
        self.original_strategy = strategy
        self.noise_scale = noise_scale
        self.projection_type = projection_type
        self.projection_rank = projection_rank
        self.resample_every_batch = resample_every_batch
        self.random_seed = random_seed
        self.device = device
        self.allow_trainable = allow_trainable
        
        # 存储原始参数和随机权重/噪声
        self.target_params: Dict[str, nn.Parameter] = {}
        self.random_weights: Dict[str, torch.Tensor] = {}  # 用于 full_random
        self.noises: Dict[str, torch.Tensor] = {}  # 用于 additive_noise
        self.forward_hooks: List = []
        self.wrapped_modules: Dict[str, nn.Module] = {}
        
        # 随机数生成器
        self.rng = torch.Generator(device=device)
        if random_seed is not None:
            self.rng.manual_seed(random_seed)
        
        # 状态标志
        self.is_enabled = True
        
        print("=" * 70)
        print(f"Random Forward Manager Initialized")
        print(f"  Strategy: {strategy}")
        print(f"  Noise Scale: {noise_scale}")
        print(f"  Allow Trainable: {allow_trainable}")
        if strategy == "full_random":
            print(f"  Projection Type: {projection_type}")
            print(f"  Projection Rank: {projection_rank if projection_rank else 'Full Rank'}")
        print(f"  Resample Every Batch: {resample_every_batch}")
        print(f"  Random Seed: {random_seed if random_seed else 'Dynamic'}")
        print("=" * 70)
    
    def register_layer(self, name: str, param: nn.Parameter):
        """注册一个参数"""
        self.target_params[name] = param
        trainable_status = "trainable" if param.requires_grad else "frozen"
        print(f"  Registered {trainable_status} param: {name:60s} shape={tuple(param.shape)}")
    
    def _generate_random_matrix(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机矩阵"""
        if self.random_seed is not None and not self.resample_every_batch:
            return torch.randn(shape, device=self.device, generator=self.rng)
        else:
            return torch.randn(shape, device=self.device)
    
    def _generate_rotation_matrix(self, n: int) -> torch.Tensor:
        """生成正交旋转矩阵"""
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
    
    def initialize_random_forward(self):
        """初始化所有参数的随机前向权重/噪声"""
        print("=" * 70)
        print("Initializing Random Forward Propagation")
        print("=" * 70)
        
        for name, param in self.target_params.items():
            if self.strategy == "additive_noise":
                # 方案A: 生成噪声
                noise = self._generate_random_matrix(param.shape)
                self.noises[name] = noise
                print(f"  [Noise] {name:50s} shape={tuple(param.shape)}")
                
            elif self.strategy == "full_random":
                # 方案B: 生成完全随机的权重
                if "weight" in name and len(param.shape) == 2:
                    out_dim, in_dim = param.shape
                    if self.projection_rank is not None:
                        proj_dim = min(self.projection_rank, in_dim, out_dim)
                    else:
                        proj_dim = in_dim
                    
                    # 生成随机权重矩阵
                    random_weight = self._generate_projection_matrix(in_dim, out_dim)
                    self.random_weights[name] = random_weight
                    print(f"  [Random Weight] {name:50s} shape={tuple(param.shape)}")
                else:
                    random_weight = self._generate_random_matrix(param.shape)
                    self.random_weights[name] = random_weight
                    print(f"  [Random] {name:50s} shape={tuple(param.shape)}")
        
        print("=" * 70)
        if self.strategy == "additive_noise":
            print(f"Total {len(self.noises)} noise tensors initialized")
        elif self.strategy == "full_random":
            print(f"Total {len(self.random_weights)} random weights initialized")
        print("=" * 70 + "\n")
    
    def disable(self):
        """关闭 random forward，恢复标准训练"""
        print("\n" + "=" * 70)
        print("DISABLING RANDOM FORWARD - Switching to Standard Training")
        print("=" * 70)
        self.is_enabled = False
        self.strategy = "none"
        
        # 移除所有 hooks
        self.remove_hooks()
        
        # 恢复所有被包装模块的原始 forward 方法
        for name, module in self.wrapped_modules.items():
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
                print(f"  Restored original forward for: {name}")
        
        print("=" * 70)
        print("Random forward disabled. Now using standard training.")
        print("=" * 70 + "\n")
    
    def enable(self):
        """重新启用 random forward"""
        if self.original_strategy == "none":
            print("Cannot enable: original strategy was 'none'")
            return
        
        print("\n" + "=" * 70)
        print("RE-ENABLING RANDOM FORWARD")
        print("=" * 70)
        self.is_enabled = True
        self.strategy = self.original_strategy
        print("Random forward re-enabled")
        print("=" * 70 + "\n")
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for handle in self.forward_hooks:
            handle.remove()
        self.forward_hooks.clear()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "strategy": self.strategy,
            "is_enabled": self.is_enabled,
            "num_target_params": len(self.target_params),
            "num_hooks": len(self.forward_hooks),
            "resample_every_batch": self.resample_every_batch,
            "allow_trainable": self.allow_trainable,
            "noise_scale": self.noise_scale,
        }
        
        if self.strategy == "full_random":
            stats["projection_type"] = self.projection_type
            stats["projection_rank"] = self.projection_rank
        
        return stats


class RandomForwardLinearAdditive(torch.autograd.Function):
    """
    方案A: Additive Noise
    Forward: output = input @ (W_real + noise_scale * noise).T
    Backward: 正常梯度，但基于带噪声的前向激活
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, noise, noise_scale, manager):
        """
        Args:
            input: (batch, in_features)
            weight: (out_features, in_features) - W_real
            bias: (out_features,)
            noise: 随机噪声张量
            noise_scale: 噪声缩放系数
            manager: RandomForwardManager 实例
        """
        # Forward 使用 W_real + noise
        noisy_weight = weight + noise_scale * noise
        output = input.matmul(noisy_weight.t())
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        # 保存 W_real 用于 backward
        ctx.save_for_backward(input, weight, bias)
        ctx.manager = manager
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        manager = ctx.manager
        
        grad_input = grad_weight = grad_bias = None
        
        # 检查 manager 是否仍然启用
        if manager and not manager.is_enabled:
            # 如果禁用，就用正常的 backward
            pass
        
        # Backward 使用 W_real (正常梯度传播)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        
        # W_real 的梯度（基于带噪声的前向激活）
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias, None, None, None


class RandomForwardLinearFull(torch.autograd.Function):
    """
    方案B: Full Random (Feedback Alignment style)
    Forward: output = input @ W_random.T
    Backward: grad_input = grad_output @ W_real (alignment)
              grad_weight = 基于 W_random 的前向激活
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, random_weight, use_random_forward, manager):
        """
        Args:
            input: (batch, in_features)
            weight: (out_features, in_features) - W_real
            bias: (out_features,)
            random_weight: 完全随机的权重
            use_random_forward: 是否使用随机前向
            manager: RandomForwardManager 实例
        """
        # 选择用于前向传播的权重
        if use_random_forward and random_weight is not None:
            forward_weight = random_weight
        else:
            forward_weight = weight
        
        # Forward 使用选定的权重
        output = input.matmul(forward_weight.t())
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        # 保存 W_real 用于 backward
        ctx.save_for_backward(input, weight, bias)
        ctx.manager = manager
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        manager = ctx.manager
        
        grad_input = grad_weight = grad_bias = None
        
        # 检查 manager 是否仍然启用
        if manager and not manager.is_enabled:
            pass
        
        # Backward 使用 W_real (Feedback Alignment)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        
        # W_real 的梯度（基于随机前向的激活）
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias, None, None, None


def wrap_layer_with_random_forward(
    module: nn.Module,
    random_manager: RandomForwardManager,
    layer_name: str
):
    """
    包装一个层，使其在 forward 时使用随机权重/噪声
    """
    if not isinstance(module, nn.Linear):
        return
    
    # 保存原始 forward 方法
    if not hasattr(module, '_original_forward'):
        module._original_forward = module.forward
        random_manager.wrapped_modules[layer_name] = module
    
    def new_forward(input):
        # 检查 manager 状态
        if not random_manager.is_enabled or random_manager.strategy == "none":
            return module._original_forward(input)
        
        # 方案A: Additive Noise
        if random_manager.strategy == "additive_noise":
            weight_name = f"{layer_name}.weight"
            noise = random_manager.noises.get(weight_name, None)
            
            # 如果需要每 batch 重新采样
            if random_manager.resample_every_batch and noise is not None:
                noise = random_manager._generate_random_matrix(module.weight.shape)
                random_manager.noises[weight_name] = noise
            
            if noise is None:
                return module._original_forward(input)
            
            # 使用自定义 autograd 函数
            output = RandomForwardLinearAdditive.apply(
                input,
                module.weight,
                module.bias,
                noise,
                random_manager.noise_scale,
                random_manager
            )
            
            return output
        
        # 方案B: Full Random
        elif random_manager.strategy == "full_random":
            weight_name = f"{layer_name}.weight"
            random_weight = random_manager.random_weights.get(weight_name, None)
            
            # 如果需要每 batch 重新采样
            if random_manager.resample_every_batch and random_weight is not None:
                if len(module.weight.shape) == 2:
                    out_dim, in_dim = module.weight.shape
                    proj_dim = random_manager.projection_rank if random_manager.projection_rank else in_dim
                    random_weight = random_manager._generate_projection_matrix(in_dim, out_dim)
                else:
                    random_weight = random_manager._generate_random_matrix(module.weight.shape)
                random_manager.random_weights[weight_name] = random_weight
            
            if random_weight is None:
                return module._original_forward(input)
            
            # 使用自定义 autograd 函数
            use_random = True
            output = RandomForwardLinearFull.apply(
                input,
                module.weight,
                module.bias,
                random_weight,
                use_random,
                random_manager
            )
            
            return output
        
        # 默认：正常前向
        return module._original_forward(input)
    
    module.forward = new_forward


def setup_random_forward_experiment(
    model,
    target_layer_names: List[str],
    strategy: str = "full_random",  # "additive_noise" or "full_random"
    noise_scale: float = 0.02,
    projection_type: str = "random",
    projection_rank: Optional[int] = None,
    resample_every_batch: bool = False,
    random_seed: Optional[int] = None,
    device: str = "cuda",
    allow_trainable: bool = False,
) -> RandomForwardManager:
    """
    设置随机前向传播实验
    
    Args:
        target_layer_names: 需要应用随机前向的层名称列表
        strategy: "additive_noise" (推荐) 或 "full_random"
        noise_scale: 噪声幅度（用于 additive_noise）
        allow_trainable: 是否允许对可训练参数应用
    """
    manager = RandomForwardManager(
        model=model,
        strategy=strategy,
        noise_scale=noise_scale,
        projection_type=projection_type,
        projection_rank=projection_rank,
        resample_every_batch=resample_every_batch,
        random_seed=random_seed,
        device=device,
        allow_trainable=allow_trainable,
    )
    
    # 注册参数
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in target_layer_names):
            # 如果 allow_trainable=True，不检查 requires_grad
            if allow_trainable or not param.requires_grad:
                manager.register_layer(name, param)
    
    # 初始化随机权重/噪声
    if strategy != "none":
        manager.initialize_random_forward()
    
    # 包装线性层
    if strategy in ["additive_noise", "full_random"]:
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in target_layer_names):
                if isinstance(module, nn.Linear):
                    wrap_layer_with_random_forward(module, manager, name)
                    print(f"  Wrapped layer: {name}")
    
    return manager


# ============ 测试代码 ============
if __name__ == "__main__":
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleModel().cuda()
    
    # 测试方案A: Additive Noise
    print("\n" + "="*70)
    print("Testing Strategy: Additive Noise")
    print("="*70)
    
    manager_a = setup_random_forward_experiment(
        model=model,
        target_layer_names=['fc1', 'fc2'],
        strategy="additive_noise",
        noise_scale=0.1,
        allow_trainable=True,
        device='cuda'
    )
    
    # 前向测试
    x = torch.randn(4, 10).cuda()
    y = torch.randint(0, 2, (4,)).cuda()
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    print(f"\nForward-Backward successful!")
    print(f"Loss: {loss.item():.4f}")
    print(f"fc1.weight.grad: {model.fc1.weight.grad.abs().mean().item():.6f}")
    
    # 测试方案B: Full Random
    print("\n" + "="*70)
    print("Testing Strategy: Full Random")
    print("="*70)
    
    model2 = SimpleModel().cuda()
    manager_b = setup_random_forward_experiment(
        model=model2,
        target_layer_names=['fc1', 'fc2'],
        strategy="full_random",
        projection_type="random",
        allow_trainable=True,
        device='cuda'
    )
    
    output2 = model2(x)
    loss2 = nn.CrossEntropyLoss()(output2, y)
    loss2.backward()
    
    print(f"\nForward-Backward successful!")
    print(f"Loss: {loss2.item():.4f}")
    print(f"fc1.weight.grad: {model2.fc1.weight.grad.abs().mean().item():.6f}")
