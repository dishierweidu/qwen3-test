# src/qwen3_omni_pretrain/parallel/tensor_parallel.py
"""
Tensor Parallel Linear Layers

Implements ColumnParallelLinear and RowParallelLinear following Megatron-LM style.

For a linear layer Y = XW + b:
- ColumnParallelLinear: W is split along columns (output dimension)
  Each GPU computes Y_i = X @ W_i, then optionally gathers results
- RowParallelLinear: W is split along rows (input dimension)
  Input X must be pre-scattered, each GPU computes Y_i = X_i @ W_i, then all-reduces

Typical usage in Transformer:
- Attention: Q, K, V projections use ColumnParallel; O projection uses RowParallel
- MLP: fc1/gate_proj/up_proj use ColumnParallel; fc2/down_proj use RowParallel
"""

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from qwen3_omni_pretrain.parallel.initialize import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


# =============================================================================
# Communication Primitives as torch.autograd.Function
# =============================================================================

class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Forward: Identity (copy input as-is)
    Backward: All-reduce gradients across TP group
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        tp_group = get_tensor_model_parallel_group()
        if tp_group is not None and get_tensor_model_parallel_world_size() > 1:
            dist.all_reduce(grad_output, group=tp_group)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Forward: All-reduce input across TP group
    Backward: Identity (pass gradient as-is)
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        tp_group = get_tensor_model_parallel_group()
        if tp_group is not None and get_tensor_model_parallel_world_size() > 1:
            dist.all_reduce(input_, group=tp_group)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """
    Forward: Scatter input along last dimension to TP group
    Backward: All-gather gradients
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return input_
        
        tp_rank = get_tensor_model_parallel_rank()
        last_dim = input_.dim() - 1
        dim_size = input_.size(last_dim)
        
        assert dim_size % tp_world_size == 0, \
            f"Last dim {dim_size} must be divisible by TP world size {tp_world_size}"
        
        chunk_size = dim_size // tp_world_size
        start = tp_rank * chunk_size
        end = start + chunk_size
        
        return input_.narrow(last_dim, start, chunk_size).contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return grad_output
        
        tp_group = get_tensor_model_parallel_group()
        last_dim = grad_output.dim() - 1
        
        # All-gather along last dimension
        grad_list = [torch.empty_like(grad_output) for _ in range(tp_world_size)]
        dist.all_gather(grad_list, grad_output.contiguous(), group=tp_group)
        
        return torch.cat(grad_list, dim=last_dim)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """
    Forward: All-gather input along last dimension from TP group
    Backward: Scatter gradients (keep only local chunk)
    """
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return input_
        
        tp_group = get_tensor_model_parallel_group()
        last_dim = input_.dim() - 1
        
        # All-gather along last dimension
        input_list = [torch.empty_like(input_) for _ in range(tp_world_size)]
        dist.all_gather(input_list, input_.contiguous(), group=tp_group)
        
        return torch.cat(input_list, dim=last_dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return grad_output
        
        tp_rank = get_tensor_model_parallel_rank()
        last_dim = grad_output.dim() - 1
        dim_size = grad_output.size(last_dim)
        
        chunk_size = dim_size // tp_world_size
        start = tp_rank * chunk_size
        
        return grad_output.narrow(last_dim, start, chunk_size).contiguous()


# Functional wrappers for communication primitives
def copy_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Copy to TP region: identity forward, all-reduce backward."""
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Reduce from TP region: all-reduce forward, identity backward."""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Scatter to TP region: scatter forward, all-gather backward."""
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Gather from TP region: all-gather forward, scatter backward."""
    return _GatherFromModelParallelRegion.apply(input_)


# =============================================================================
# Parallel Linear Layers
# =============================================================================

def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1,
):
    """
    Initialize affine weight for a parallel layer.
    
    Args:
        weight: The weight tensor to initialize (already partitioned)
        out_features: Total output features (before partitioning)
        in_features: Total input features (before partitioning)
        per_partition_size: Size of each partition
        partition_dim: Which dimension is partitioned (0 for rows, 1 for columns)
        init_method: Initialization function
        stride: Stride for interleaved partitioning (usually 1)
    """
    # Set random seed for reproducibility
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    
    # Initialize master weight on CPU to ensure consistency
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype)
    init_method(master_weight)
    
    # Extract local partition
    if partition_dim == 0:
        # Row partition (RowParallel): split along out_features
        start = tp_rank * per_partition_size
        end = start + per_partition_size
        weight.data.copy_(master_weight[start:end, :])
    else:
        # Column partition (ColumnParallel): split along in_features (but weight is [out, in])
        # Actually for ColumnParallel, we split output dim, so partition_dim=0 in weight
        start = tp_rank * per_partition_size
        end = start + per_partition_size
        weight.data.copy_(master_weight[start:end, :])


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism (output dimension split across TP group).
    
    Y = XW + b where W has shape [in_features, out_features]
    W is split along out_features: each GPU stores W[:, out_features/tp_size]
    
    Forward pass:
        1. Input X is copied to all GPUs (identity or all-reduce in backward)
        2. Each GPU computes Y_local = X @ W_local
        3. If gather_output=True, all-gather Y across TP group
        
    Args:
        in_features: Total input features
        out_features: Total output features (will be split across TP group)
        bias: Whether to include bias
        gather_output: If True, all-gather output; if False, keep partial output
        init_method: Weight initialization function
        stride: Stride for interleaved partitioning
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
        stride: int = 1,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        
        tp_world_size = get_tensor_model_parallel_world_size()
        
        assert out_features % tp_world_size == 0, \
            f"out_features ({out_features}) must be divisible by TP world size ({tp_world_size})"
        
        self.output_size_per_partition = out_features // tp_world_size
        
        # Weight: [out_features_per_partition, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
        
        # Initialize weight
        if init_method is None:
            init_method = lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        
        self._init_weight(init_method, stride)
        
        # Initialize bias
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _init_weight(self, init_method: Callable, stride: int = 1):
        """Initialize weight with proper partitioning."""
        _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            partition_dim=0,  # split output dim
            init_method=init_method,
            stride=stride,
        )
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features] if gather_output=True,
            else [..., out_features // tp_size]
        """
        # Copy input to TP region (identity forward, all-reduce backward)
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        
        # Local linear
        output_parallel = F.linear(input_parallel, self.weight, 
                                   self.bias if not self.skip_bias_add else None)
        
        # Gather output if needed
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
            # Handle bias for gathered output
            if self.skip_bias_add and self.bias is not None:
                # Gather bias as well
                bias = gather_from_tensor_model_parallel_region(
                    self.bias.unsqueeze(0)
                ).squeeze(0)
                return output, bias
            return output
        else:
            if self.skip_bias_add and self.bias is not None:
                return output_parallel, self.bias
            return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism (input dimension split across TP group).
    
    Y = XW + b where W has shape [in_features, out_features]
    W is split along in_features: each GPU stores W[in_features/tp_size, :]
    
    Forward pass:
        1. Input X must already be split across TP group
        2. Each GPU computes Y_local = X_local @ W_local
        3. All-reduce Y across TP group to get final output
        
    Args:
        in_features: Total input features (will be split across TP group)
        out_features: Total output features
        bias: Whether to include bias
        input_is_parallel: If True, input is already split; if False, scatter input
        init_method: Weight initialization function
        stride: Stride for interleaved partitioning
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        init_method: Optional[Callable] = None,
        stride: int = 1,
        skip_bias_add: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        
        tp_world_size = get_tensor_model_parallel_world_size()
        
        assert in_features % tp_world_size == 0, \
            f"in_features ({in_features}) must be divisible by TP world size ({tp_world_size})"
        
        self.input_size_per_partition = in_features // tp_world_size
        
        # Weight: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.input_size_per_partition)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        # Initialize weight
        if init_method is None:
            init_method = lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        
        self._init_weight(init_method, stride)
        
        # Initialize bias
        if self.bias is not None:
            fan_in = in_features  # Use full in_features for bias init
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _init_weight(self, init_method: Callable, stride: int = 1):
        """Initialize weight with proper partitioning."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        
        # Initialize master weight
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype)
        init_method(master_weight)
        
        # Extract local partition (along columns of master weight)
        start = tp_rank * self.input_size_per_partition
        end = start + self.input_size_per_partition
        self.weight.data.copy_(master_weight[:, start:end])
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_: If input_is_parallel=True, shape [..., in_features // tp_size]
                   If input_is_parallel=False, shape [..., in_features]
                   
        Returns:
            Output tensor of shape [..., out_features]
        """
        # Scatter input if needed
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        
        # Local linear (without bias)
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce to combine partial results
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        # Add bias
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias
        
        if self.skip_bias_add:
            return output, self.bias
        return output


# =============================================================================
# Utility functions for converting standard layers to parallel layers
# =============================================================================

def convert_linear_to_column_parallel(
    linear: nn.Linear,
    gather_output: bool = True,
) -> ColumnParallelLinear:
    """
    Convert a standard nn.Linear to ColumnParallelLinear.
    
    This is useful for initializing from pretrained weights.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    
    in_features = linear.in_features
    out_features = linear.out_features
    has_bias = linear.bias is not None
    
    parallel_linear = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
        gather_output=gather_output,
    )
    
    # Copy partitioned weight
    per_partition = out_features // tp_world_size
    start = tp_rank * per_partition
    end = start + per_partition
    
    with torch.no_grad():
        parallel_linear.weight.copy_(linear.weight[start:end, :])
        if has_bias:
            parallel_linear.bias.copy_(linear.bias[start:end])
    
    return parallel_linear


def convert_linear_to_row_parallel(
    linear: nn.Linear,
    input_is_parallel: bool = True,
) -> RowParallelLinear:
    """
    Convert a standard nn.Linear to RowParallelLinear.
    
    This is useful for initializing from pretrained weights.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    
    in_features = linear.in_features
    out_features = linear.out_features
    has_bias = linear.bias is not None
    
    parallel_linear = RowParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
        input_is_parallel=input_is_parallel,
    )
    
    # Copy partitioned weight
    per_partition = in_features // tp_world_size
    start = tp_rank * per_partition
    end = start + per_partition
    
    with torch.no_grad():
        parallel_linear.weight.copy_(linear.weight[:, start:end])
        if has_bias:
            # Bias is not partitioned for RowParallel
            parallel_linear.bias.copy_(linear.bias)
    
    return parallel_linear
