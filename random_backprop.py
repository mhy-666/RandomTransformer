
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Backpropagation Experiment
支持两种方案：
1. 方案A：完全替换反向传播中的参数为随机参数
2. 方案B：使用降维的随机投影矩阵（支持随机投影或旋转矩阵）
"""
 
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class RandomBackpropManager:
    """管理随机反向传播的核心类"""

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "none",  # "none", "full_random", "low_rank_projection"
        projection_type: str = "random",  # "random" or "rotation"
        projection_rank: Optional[int] = None,  # 投影矩阵的秩（仅用于low_rank_projection）
        resample_every_batch: bool = False,  # 是否每个batch重新采样随机参数
        random_seed: Optional[int] = None,  # 固定随机种子
        device: str = "cuda",
        allow_trainable: bool = False,
    ):
        """
        Args:
            model: 要修改的PyTorch模型
            strategy: 反向传播策略
                - "none": 正常反向传播（baseline）
                - "full_random": 方案A，完全随机替换
                - "low_rank_projection": 方案B，低秩随机投影
            projection_type: 投影矩阵类型（仅用于方案B）
                - "random": 随机高斯投影
                - "rotation": 正交旋转矩阵
            projection_rank: 投影矩阵的秩，None表示使用原始维度
            resample_every_batch: 是否每个batch重新生成随机参数
            random_seed: 固定随机种子以保证可复现性
            device: 设备
        """
        self.model = model
        self.strategy = strategy
        self.projection_type = projection_type
        self.projection_rank = projection_rank
        self.resample_every_batch = resample_every_batch
        self.random_seed = random_seed
        self.device = device

        # 存储原始参数和对应的随机替代
        self.frozen_params: Dict[str, nn.Parameter] = {}
        self.random_replacements: Dict[str, torch.Tensor] = {}
        self.backward_hooks: List = []

        # 随机数生成器
        self.rng = torch.Generator(device=device)
        if random_seed is not None:
            self.rng.manual_seed(random_seed)

        print("=" * 70)
        print(f"Random Backpropagation Manager Initialized")
        print(f"  Strategy: {strategy}")
        if strategy == "low_rank_projection":
            print(f"  Projection Type: {projection_type}")
            print(f"  Projection Rank: {projection_rank if projection_rank else 'Full Rank'}")
        print(f"  Resample Every Batch: {resample_every_batch}")
        print(f"  Random Seed: {random_seed if random_seed else 'Dynamic'}")
        print("=" * 70)

    def register_frozen_layer(self, name: str, param: nn.Parameter):
        """注册一个冻结的参数"""
        self.frozen_params[name] = param
        print(f"  Registered frozen param: {name:60s} shape={tuple(param.shape)}")

    def _generate_random_matrix(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机矩阵"""
        if self.random_seed is not None and not self.resample_every_batch:
            # 使用固定种子
            return torch.randn(shape, device=self.device, generator=self.rng) * 0.02
        else:
            # 动态生成
            return torch.randn(shape, device=self.device) * 0.02

    def _generate_rotation_matrix(self, n: int) -> torch.Tensor:
        """生成n×n的正交旋转矩阵（使用QR分解）"""
        # 生成随机矩阵
        if self.random_seed is not None and not self.resample_every_batch:
            A = torch.randn(n, n, device=self.device, generator=self.rng)
        else:
            A = torch.randn(n, n, device=self.device)

        # QR分解得到正交矩阵
        Q, R = torch.linalg.qr(A)
        # 调整符号以确保R的对角线为正（标准化）
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        return Q

    def _generate_projection_matrix(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """
        生成投影矩阵（用于方案B）

        Args:
            in_dim: 输入维度
            out_dim: 输出维度

        Returns:
            projection matrix of shape (out_dim, in_dim)
        """
        if self.projection_type == "rotation":
            # 旋转矩阵必须是方阵
            if in_dim == out_dim:
                return self._generate_rotation_matrix(in_dim)
            else:
                # 如果维度不匹配，先生成方阵再截取
                max_dim = max(in_dim, out_dim)
                Q = self._generate_rotation_matrix(max_dim)
                return Q[:out_dim, :in_dim]
        else:  # random projection
            # 标准高斯随机投影
            proj = self._generate_random_matrix((out_dim, in_dim))
            # 归一化以保持梯度尺度
            proj = proj / np.sqrt(in_dim)
            return proj

    def initialize_random_replacements(self):
        """初始化所有冻结参数的随机替代"""
        print("=" * 70)
        print("Initializing Random Replacements for Backpropagation")
        print("=" * 70)

        for name, param in self.frozen_params.items():
            if self.strategy == "full_random":
                # 方案A：完全随机替换，保持相同形状
                random_param = self._generate_random_matrix(param.shape)
                self.random_replacements[name] = random_param
                print(f"  [Full Random] {name:50s} shape={tuple(param.shape)}")

            elif self.strategy == "low_rank_projection":
                # 方案B：低秩投影
                # 只对weight矩阵进行投影，bias保持不变
                if "weight" in name and len(param.shape) == 2:
                    out_dim, in_dim = param.shape

                    # 确定投影后的维度
                    if self.projection_rank is not None:
                        proj_dim = min(self.projection_rank, in_dim, out_dim)
                    else:
                        proj_dim = in_dim

                    # 生成投影矩阵 P: (out_dim, proj_dim)
                    proj_matrix = self._generate_projection_matrix(in_dim, proj_dim)

                    self.random_replacements[name] = proj_matrix
                    print(f"  [Projection] {name:50s} original={tuple(param.shape)} -> proj={(proj_dim, in_dim)}")
                else:
                    # bias或非2D参数，使用原始形状的随机矩阵
                    random_param = self._generate_random_matrix(param.shape)
                    self.random_replacements[name] = random_param
                    print(f"  [Random] {name:50s} shape={tuple(param.shape)}")

        print("=" * 70)
        print(f"Total {len(self.random_replacements)} random replacements initialized")
        print("=" * 70 + "")

    def _create_backward_hook_full_random(self, param_name: str, original_param: nn.Parameter):
        """创建方案A的backward hook：完全随机替换"""
        def hook(grad):
            if grad is None:
                return grad

            # 如果需要每个batch重新采样
            if self.resample_every_batch:
                random_replacement = self._generate_random_matrix(original_param.shape)
                self.random_replacements[param_name] = random_replacement

            # 注意：这里不直接修改grad，而是通过hook在梯度计算时影响
            # 实际上需要在forward时就替换参数，但这会影响forward结果
            # 更好的方式是使用custom autograd function

            return grad

        return hook

    def _create_backward_hook_projection(self, param_name: str, original_param: nn.Parameter):
        """创建方案B的backward hook：投影梯度"""
        def hook(grad):
            if grad is None:
                return grad

            # 如果需要每个batch重新采样投影矩阵
            if self.resample_every_batch:
                if "weight" in param_name and len(original_param.shape) == 2:
                    out_dim, in_dim = original_param.shape
                    proj_dim = self.projection_rank if self.projection_rank else in_dim
                    proj_matrix = self._generate_projection_matrix(in_dim, min(proj_dim, in_dim, out_dim))
                    self.random_replacements[param_name] = proj_matrix

            return grad

        return hook

    def apply_hooks(self):
        """应用backward hooks到冻结的参数上"""
        if self.strategy == "none":
            print("Strategy is 'none', no hooks applied.")
            return

        print("=" * 70)
        print(f"Applying Backward Hooks (Strategy: {self.strategy})")
        print("=" * 70)

        for name, param in self.frozen_params.items():
            if self.strategy == "full_random":
                hook = self._create_backward_hook_full_random(name, param)
            elif self.strategy == "low_rank_projection":
                hook = self._create_backward_hook_projection(name, param)
            else:
                continue

            # 注册hook
            handle = param.register_hook(hook)
            self.backward_hooks.append(handle)
            print(f"  Hook registered: {name}")

        print(f"Total {len(self.backward_hooks)} hooks applied")
        print("=" * 70 + "")

    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.backward_hooks:
            handle.remove()
        self.backward_hooks.clear()
        print("All backward hooks removed.")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "strategy": self.strategy,
            "num_frozen_params": len(self.frozen_params),
            "num_hooks": len(self.backward_hooks),
            "resample_every_batch": self.resample_every_batch,
        }

        if self.strategy == "low_rank_projection":
            stats["projection_type"] = self.projection_type
            stats["projection_rank"] = self.projection_rank

        return stats


class RandomBackwardLinear(torch.autograd.Function):
    """
    自定义autograd函数，用于实现随机反向传播
    这是更精确的实现方式，直接控制forward和backward的行为
    """

    @staticmethod
    def forward(ctx, input, weight, bias, random_weight, use_random_backward):
        """
        Forward使用真实权重

        Args:
            input: 输入tensor (batch, in_features)
            weight: 真实权重 (out_features, in_features)
            bias: 偏置 (out_features,)
            random_weight: 用于backward的随机权重
            use_random_backward: 是否在backward时使用随机权重
        """
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # 保存用于backward的内容
        ctx.save_for_backward(input, weight, bias, random_weight)
        ctx.use_random_backward = use_random_backward

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward可选使用随机权重
        """
        input, weight, bias, random_weight = ctx.saved_tensors
        use_random_backward = ctx.use_random_backward

        grad_input = grad_weight = grad_bias = None

        # 选择用于梯度传播的权重
        backward_weight = random_weight if (use_random_backward and random_weight is not None) else weight

        if ctx.needs_input_grad[0]:
            # 计算输入的梯度：使用随机权重或真实权重
            grad_input = grad_output.matmul(backward_weight)

        # 注意：weight和bias的梯度不计算（因为它们是frozen的）
        # grad_weight 和 grad_bias 保持为 None

        return grad_input, grad_weight, grad_bias, None, None


def wrap_frozen_layer_with_random_backward(
    module: nn.Module,
    random_manager: RandomBackpropManager,
    layer_name: str
):
    """
    包装一个frozen层，使其在backward时使用随机参数

    这个函数会修改module的forward方法
    """
    if not isinstance(module, nn.Linear):
        return  # 目前只支持Linear层

    original_forward = module.forward

    def new_forward(input):
        if random_manager.strategy == "none":
            return original_forward(input)

        # 获取随机替换参数
        weight_name = f"{layer_name}.weight"
        random_weight = random_manager.random_replacements.get(weight_name, None)

        if random_weight is None:
            # 如果没有随机参数，使用正常forward
            return original_forward(input)

        # 使用自定义autograd函数
        use_random = (random_manager.strategy in ["full_random", "low_rank_projection"])
        output = RandomBackwardLinear.apply(
            input,
            module.weight,
            module.bias,
            random_weight,
            use_random
        )

        return output

    # 替换forward方法
    module.forward = new_forward


# ==================== 集成到你的训练代码中 ====================

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
        model: GPT2模型
        frozen_layer_names: 冻结的层名称列表
        strategy: "none", "full_random", "low_rank_projection"
        projection_type: "random" or "rotation"
        projection_rank: 投影秩
        resample_every_batch: 是否每batch重新采样
        random_seed: 随机种子
        device: 设备

    Returns:
        RandomBackpropManager实例
    """
    # 创建管理器
    manager = RandomBackpropManager(
        model=model,
        strategy=strategy,
        projection_type=projection_type,
        projection_rank=projection_rank,
        resample_every_batch=resample_every_batch,
        random_seed=random_seed,
        device=device,
    )

    # 注册冻结的参数
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in frozen_layer_names):
            if not param.requires_grad:  # 确保是冻结的
                manager.register_frozen_layer(name, param)

    # 初始化随机替换
    if strategy != "none":
        manager.initialize_random_replacements()

    # 包装冻结的线性层（使用custom autograd function）
    if strategy in ["full_random", "low_rank_projection"]:
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in frozen_layer_names):
                if isinstance(module, nn.Linear):
                    wrap_frozen_layer_with_random_backward(model, manager, name)
                    print(f"Wrapped layer with random backward: {name}")

    return manager


# ==================== 使用示例 ====================

def example_usage():
    """示例：如何在你的训练代码中使用"""

    # 假设你已经有了model和frozen的层
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # frozen_layer_names = ["transformer.h.0", "transformer.h.1", ...]

    # 方案A：完全随机替换
    manager_a = setup_random_backprop_experiment(
        model=None,  # 你的model
        frozen_layer_names=["transformer.h.0.attn", "transformer.h.0.mlp"],
        strategy="full_random",
        resample_every_batch=False,  # 固定随机参数
        random_seed=42,
        device="cuda"
    )

    # 方案B：低秩随机投影
    manager_b = setup_random_backprop_experiment(
        model=None,  # 你的model
        frozen_layer_names=["transformer.h.0.attn", "transformer.h.0.mlp"],
        strategy="low_rank_projection",
        projection_type="rotation",  # 或 "random"
        projection_rank=256,  # 投影到256维
        resample_every_batch=False,
        random_seed=42,
        device="cuda"
    )

    # Baseline：正常训练（不使用随机反向传播）
    manager_baseline = setup_random_backprop_experiment(
        model=None,
        frozen_layer_names=[],
        strategy="none",
        device="cuda"
    )

    print("\nExample setup completed!")
    print("In your training loop, the backward pass will automatically use random parameters.")


if __name__ == "__main__":
    print("Random Backpropagation Implementation")
    print("=" * 70)
    example_usage()
