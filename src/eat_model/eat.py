"""
EAT: Endoscopic Adaptive Transformer for Enhanced Polyp Segmentation.

This module implements the EAT (Endoscopic Adaptive Transformer) architecture,
a cutting-edge framework specifically engineered for the nuanced task of polyp
segmentation in endoscopic imaging.

Key Features:
- Pyramid Vision Transformer (PVT v2) backbone for hierarchical multi-scale
  representation extraction
- Adaptive Perception Module (APM) with adaptive perceptive-field mechanism
  to dynamically capture both fine-grained local details and broad contextual
  information
- Deformable Convolution Networks v4 (DCNv4) for flexible and responsive
  representation extraction
- Comprehensive Feature Extractor (CFE) for enriching feature maps with
  both fine-grained local details and global contextual information

The framework addresses the inherent morphological variability of polyps,
which presents significant challenges to existing segmentation methodologies.
Unlike conventional approaches that generate feature maps by merging regular
image patches with convolutional units into fixed local windows, EAT leverages
an adaptive perceptive-field mechanism that enables hierarchical representation
learning to dynamically capture critical local features.

Published in IEEE Transactions on Medical Imaging (TMI).
DOI: 10.1109/TMI.2025.3615677

Reference:
    Pang Y, Long Y, Chen Z, et al. "Endoscopic Adaptive Transformer for
    Enhanced Polyp Segmentation in Endoscopic Imaging." IEEE Transactions
    on Medical Imaging, 2025.

Code: https://github.com/deepang-ai/EAT
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast

from timm.models.layers import DropPath, make_divisible, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Type aliases for improved readability
TensorPair = Tuple[Tensor, Tensor]
TensorQuad = Tuple[Tensor, Tensor, Tensor, Tensor]
FeatureList = List[Tensor]


def _import_dcnv4() -> Type:
    """
    Lazily import DCNv4 module with graceful fallback.
    
    Returns:
        DCNv4 class from either installed package or local repository.
        
    Raises:
        ImportError: If DCNv4 cannot be imported from any source.
    """
    try:
        from DCNv4.modules import DCNv4
        logger.debug("DCNv4 imported from installed package")
        return DCNv4
    except ImportError:
        pass
    
    try:
        from DCNv4_op.DCNv4.modules import DCNv4
        logger.debug("DCNv4 imported from local repository")
        return DCNv4
    except ImportError as e:
        raise ImportError(
            "DCNv4 is required but could not be imported. "
            "Please build/install it from DCNv4_op/ directory "
            "(see DCNv4_op/make.sh for instructions)."
        ) from e


def _import_flash_attention() -> Tuple[Optional[Callable], bool]:
    """
    Attempt to import Flash Attention with availability check.
    
    Returns:
        Tuple of (flash_attn_func or None, availability boolean).
    """
    try:
        from flash_attn import flash_attn_func
        logger.info("Flash Attention is available and will be used for acceleration")
        return flash_attn_func, True
    except ImportError:
        logger.warning(
            "Flash Attention not available. Falling back to standard attention. "
            "Install flash-attn for improved performance: pip install flash-attn"
        )
        return None, False


# Lazy imports for optional dependencies
DCNv4 = _import_dcnv4()
flash_attn_func, _FLASH_ATTN_AVAILABLE = _import_flash_attention()


class InitMethod(Enum):
    """Weight initialization methods."""
    TRUNC_NORMAL = auto()
    XAVIER_UNIFORM = auto()
    KAIMING_NORMAL = auto()
    DEFAULT = auto()


@dataclass
class ModelConfig:
    """
    Configuration dataclass for EAT model hyperparameters.
    
    The EAT framework processes input endoscopic images through a PVT backbone,
    extracting multi-scale feature representations {f1, f2, f3, f4} at four
    hierarchical levels. Lower levels (f1, f2) capture fine-grained features
    like edges and textures, while higher levels (f3, f4) capture abstract
    high-level semantic information.
    
    Attributes:
        in_channels: Number of input image channels (C).
        out_channels: Number of output segmentation classes (J).
        dims: Feature dimensions at each pyramid stage [C·2^5, C·2^6, C·2^7, C·2^8].
        out_dim: Output dimension for intermediate features (default: 32).
        kernel_size: Kernel size for convolution operations.
        mlp_ratio: Expansion ratio for MLP layers.
        gla_os: Offset scales (s) for deformable convolution in [APU, Bypass].
                Controls the magnitude of learnable offsets Δpk.
                Optimal: APM=3.0, shallow-level=1.0 (per ablation study).
        l_feature_v4: Whether to use DCNv4 for shallow-level feature (f1) processing.
        l_feature_os: Offset scale for shallow-level deformable convolution.
        pretrained_path: Path to pretrained PVT backbone weights.
        drop_path_rate: Stochastic depth rate for regularization.
        use_checkpoint: Whether to use gradient checkpointing for memory efficiency.
    """
    in_channels: int = 3
    out_channels: int = 1
    dims: Tuple[int, ...] = (64, 128, 320, 512)
    out_dim: int = 32
    kernel_size: int = 3
    mlp_ratio: int = 4
    gla_os: Tuple[float, float] = (2.5, 1.0)
    l_feature_v4: bool = True
    l_feature_os: float = 2.0
    pretrained_path: Optional[str] = "./pretrained/pvt_v2_b2.pth"
    drop_path_rate: float = 0.1
    use_checkpoint: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.out_channels > 0, "out_channels must be positive"
        assert len(self.dims) == 4, "dims must have exactly 4 stages"
        assert all(d > 0 for d in self.dims), "all dims must be positive"
        assert self.out_dim > 0, "out_dim must be positive"
        assert self.kernel_size > 0 and self.kernel_size % 2 == 1, \
            "kernel_size must be positive and odd"
        assert self.mlp_ratio > 0, "mlp_ratio must be positive"
        assert 0 <= self.drop_path_rate < 1, "drop_path_rate must be in [0, 1)"


@dataclass  
class PVTConfig:
    """
    Configuration for Pyramid Vision Transformer backbone.
    
    Attributes:
        img_size: Input image size (assumes square).
        patch_size: Patch embedding patch size.
        embed_dims: Embedding dimensions at each stage.
        num_heads: Number of attention heads at each stage.
        mlp_ratios: MLP expansion ratios at each stage.
        sr_ratios: Spatial reduction ratios at each stage.
        depths: Number of transformer blocks at each stage.
        qkv_bias: Whether to use bias in QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
    """
    img_size: int = 224
    patch_size: int = 4
    embed_dims: Tuple[int, ...] = (64, 128, 320, 512)
    num_heads: Tuple[int, ...] = (1, 2, 5, 8)
    mlp_ratios: Tuple[int, ...] = (8, 8, 4, 4)
    sr_ratios: Tuple[int, ...] = (8, 4, 2, 1)
    depths: Tuple[int, ...] = (3, 4, 6, 3)
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


# =============================================================================
# Weight Initialization Utilities
# =============================================================================

def _init_weights_trunc_normal(module: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights using truncated normal distribution.
    
    Args:
        module: PyTorch module to initialize.
        std: Standard deviation for truncated normal.
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


def _init_weights_xavier(module: nn.Module) -> None:
    """
    Initialize weights using Xavier uniform distribution.
    
    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)


# =============================================================================
# Core Building Blocks
# =============================================================================

class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding layer for Vision Transformers.
    
    Converts images to patch embeddings with overlapping patches,
    which helps preserve local continuity between patches.
    
    Args:
        img_size: Input image size (assumes square).
        patch_size: Size of each patch.
        stride: Stride for patch extraction.
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, N, embed_dim), H_out, W_out
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(partial(_init_weights_trunc_normal, std=0.02))

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        """
        Forward pass for patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Tuple of (embedded patches, height, width).
        """
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class DWConv(nn.Module):
    """
    Depth-wise Convolution module for positional encoding.
    
    Applies depth-wise separable convolution to add positional information
    to the feature maps in a parameter-efficient manner.
    
    Args:
        dim: Number of input/output channels.
        
    Shape:
        - Input: (B, N, C) with H, W as additional parameters
        - Output: (B, N, C)
    """
    
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Apply depth-wise convolution with shape transformation.
        
        Args:
            x: Input tensor of shape (B, N, C).
            H: Height of the feature map.
            W: Width of the feature map.
            
        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Spatial Reduction Attention (SRA) module.
    
    Implements efficient self-attention with optional spatial reduction
    for handling high-resolution feature maps.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
        qk_scale: Custom scale for attention scores.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        sr_ratio: Spatial reduction ratio for keys and values.
        
    Shape:
        - Input: (B, N, C) with H, W as additional parameters
        - Output: (B, N, C)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(partial(_init_weights_trunc_normal, std=0.02))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Compute spatial reduction attention.
        
        Args:
            x: Input tensor of shape (B, N, C).
            H: Height of the feature map.
            W: Width of the feature map.
            
        Returns:
            Attention output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FlashAttention(nn.Module):
    """
    Flash Attention module with automatic fallback.
    
    Implements efficient attention using Flash Attention when available,
    with automatic fallback to standard attention for compatibility.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
        qk_scale: Custom scale for attention scores.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        sr_ratio: Spatial reduction ratio for keys and values.
        
    Shape:
        - Input: (B, N, C) with H, W as additional parameters
        - Output: (B, N, C)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(_init_weights_xavier)

    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        """
        Standard attention computation fallback.
        
        Args:
            q: Query tensor of shape (B, num_heads, N, head_dim).
            k: Key tensor of shape (B, num_heads, M, head_dim).
            v: Value tensor of shape (B, num_heads, M, head_dim).
            
        Returns:
            Attention output tensor.
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Compute attention using Flash Attention or standard attention.
        
        Args:
            x: Input tensor of shape (B, N, C).
            H: Height of the feature map.
            W: Width of the feature map.
            
        Returns:
            Attention output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv.unbind(0)

        if _FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
            # Flash Attention requires specific tensor layout: (B, N, num_heads, head_dim)
            q_flash = q.transpose(1, 2).to(torch.bfloat16)
            k_flash = k.transpose(1, 2).to(torch.bfloat16)
            v_flash = v.transpose(1, 2).to(torch.bfloat16)
            
            attn_output = flash_attn_func(q_flash, k_flash, v_flash, softmax_scale=self.scale)
            attn_output = attn_output.reshape(B, N, self.dim).to(x.dtype)
        else:
            attn_output = self._standard_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2).reshape(B, N, self.dim)

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron with depth-wise convolution.
    
    Implements a feed-forward network with GELU activation and
    depth-wise convolution for positional encoding.
    
    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension.
        out_features: Output feature dimension.
        act_layer: Activation layer class.
        drop: Dropout rate.
        
    Shape:
        - Input: (B, N, in_features) with H, W as additional parameters
        - Output: (B, N, out_features)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(partial(_init_weights_trunc_normal, std=0.02))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (B, N, in_features).
            H: Height of the feature map.
            W: Width of the feature map.
            
        Returns:
            Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    
    Implements a standard transformer block with pre-normalization,
    attention, and feed-forward network with residual connections.
    
    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension expansion ratio.
        qkv_bias: Whether to use bias in QKV projections.
        qk_scale: Custom scale for attention scores.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        act_layer: Activation layer class.
        norm_layer: Normalization layer class.
        sr_ratio: Spatial reduction ratio.
        use_flash_attn: Whether to use Flash Attention.
        
    Shape:
        - Input: (B, N, dim) with H, W as additional parameters
        - Output: (B, N, dim)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        sr_ratio: int = 1,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Select attention implementation
        attn_cls = FlashAttention if use_flash_attn else Attention
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(partial(_init_weights_trunc_normal, std=0.02))

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (B, N, dim).
            H: Height of the feature map.
            W: Width of the feature map.
            
        Returns:
            Output tensor of shape (B, N, dim).
        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# =============================================================================
# Pyramid Vision Transformer Backbone
# =============================================================================

class PyramidVisionTransformerImpr(nn.Module):
    """
    Improved Pyramid Vision Transformer (PVT v2) backbone.
    
    A hierarchical vision transformer that processes images at multiple scales,
    producing multi-scale feature maps suitable for dense prediction tasks.
    
    Args:
        img_size: Input image size.
        patch_size: Patch embedding patch size.
        in_chans: Number of input channels.
        num_classes: Number of classification classes.
        embed_dims: Embedding dimensions at each stage.
        num_heads: Number of attention heads at each stage.
        mlp_ratios: MLP expansion ratios at each stage.
        qkv_bias: Whether to use bias in QKV projections.
        qk_scale: Custom scale for attention scores.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer class.
        depths: Number of transformer blocks at each stage.
        sr_ratios: Spatial reduction ratios at each stage.
        
    Shape:
        - Input: (B, in_chans, H, W)
        - Output: List of 4 tensors with shapes (B, embed_dims[i], H_i, W_i)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        mlp_ratios: Sequence[int] = (4, 4, 4, 4),
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        depths: Sequence[int] = (3, 4, 6, 3),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)

        # Patch embedding layers for each stage
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Build transformer blocks for each stage
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0],
            )
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1],
            )
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2],
            )
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3],
            )
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(partial(_init_weights_trunc_normal, std=0.02))

    def reset_drop_path(self, drop_path_rate: float) -> None:
        """
        Reset drop path rates across all blocks.
        
        Args:
            drop_path_rate: New maximum drop path rate.
        """
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self) -> None:
        """Freeze patch embedding layers for fine-tuning."""
        self.patch_embed1.requires_grad_(False)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        """Return parameter names that should not have weight decay."""
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def forward_features(self, x: Tensor) -> FeatureList:
        """
        Extract multi-scale features from input.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            List of 4 feature tensors at different scales.
        """
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x: Tensor) -> FeatureList:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            List of 4 feature tensors at different scales.
        """
        return self.forward_features(x)


@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    """
    PVT-v2-B2 model variant.
    
    A medium-sized PVT v2 model with ~25M parameters,
    suitable for various vision tasks.
    """
    
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Sequence[int] = (64, 128, 320, 512),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_chans=in_chans,
            patch_size=4,
            embed_dims=list(embed_dims),
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        )


# =============================================================================
# Activation Functions
# =============================================================================

class SwishImplementation(torch.autograd.Function):
    """
    Memory-efficient Swish activation implementation.
    
    Uses custom backward pass to reduce memory usage during training.
    """
    
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Swish(nn.Module):
    """
    Swish activation function module.
    
    Swish(x) = x * sigmoid(x)
    
    Provides smoother gradients compared to ReLU and has been shown
    to improve performance in various deep learning tasks.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        return SwishImplementation.apply(x)


# =============================================================================
# MLP and Convolution Modules
# =============================================================================

class MLP(nn.Module):
    """
    Simple MLP with 1x1 convolutions.
    
    Two-layer perceptron using 1x1 convolutions for channel mixing.
    
    Args:
        dim: Input/output dimension.
        mlp_ratio: Hidden dimension expansion ratio.
        act: Activation function.
        
    Shape:
        - Input: (B, dim, H, W)
        - Output: (B, dim, H, W)
    """
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.line_conv_0 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1, bias=False)
        self.act = act
        self.line_conv_1 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x


class BasicConv2d(nn.Module):
    """
    Basic convolution block with BatchNorm and activation.
    
    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        act: Activation function.
        
    Shape:
        - Input: (B, in_planes, H, W)
        - Output: (B, out_planes, H', W')
    """
    
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        act: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DFConv(nn.Module):
    """
    Deformable Feature Convolution (DFConv) using DCNv4.
    
    This module applies deformable convolution with learnable offsets for
    adaptive spatial sampling. In the EAT architecture, DFConv is used to
    process the shallowest features (f_1) from the PVT-v2 backbone to enhance
    edge detection and preserve fine spatial details.
    
    The deformable convolution follows Equation 3 from the paper:
        ``f''[p_0] = Σ_k w[k] · f[p_0 + k + s · Δp_k]``
    
    where:
        - p_0: Center position of the sampling window
        - k: Enumeration of kernel sampling positions
        - w[k]: Weight at position k
        - s: Offset scale controlling the magnitude of Δp_k
        - Δp_k: Learnable offset at position k
    
    According to Table VI in the paper, the optimal offset scale for
    shallow-level feature processing is 1.0.
    
    Reference:
        IEEE TMI Paper, Section III-A (shallow feature processing)
    
    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        act: Activation function.
        offset_scale: Scale factor (s) for deformable offsets.
            Controls the magnitude of learnable offsets Δp_k.
            Default: 2.0. Paper optimal: 1.0 (Table VI).
        
    Shape:
        - Input: (B, in_planes, H, W)
        - Output: (B, out_planes, H, W)
    """
    
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        act: nn.Module = nn.ReLU(inplace=True),
        offset_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.conv = DCNv4(
            in_planes,
            kernel_size=3,
            stride=1,
            pad=1,
            group=in_planes // out_planes,
            offset_scale=offset_scale,
            bias=False,
            layer_scale=1.0,
            dw_kernel_size=3,
            mlp_ratio=4.0,
            drop_path_rate=0.4,
            norm_layer='LN',
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B, H * W, C)
        x = self.conv(x)
        x = x.reshape(B, C, H, W)
        
        # Channel reduction via averaging
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) * 0.5
        
        return self.act(self.bn(x))

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module.
    
    Implements channel attention by modeling inter-channel dependencies
    through a squeeze-and-excitation mechanism.
    
    Args:
        channels: Number of input/output channels.
        rd_ratio: Reduction ratio for the bottleneck.
        rd_channels: Explicit reduced channel count (overrides rd_ratio).
        rd_divisor: Divisor for making rd_channels divisible.
        add_maxpool: Whether to add max pooling alongside avg pooling.
        bias: Whether to use bias in FC layers.
        act: Activation function.
        norm_layer: Optional normalization layer.
        gate_layer: Gating activation (typically Sigmoid).
        
    Shape:
        - Input: (B, channels, H, W)
        - Output: (B, channels, H, W)
    """
    
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1.0 / 16,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        add_maxpool: bool = False,
        bias: bool = True,
        act: nn.Module = nn.GELU(),
        norm_layer: Optional[Type[nn.Module]] = None,
        gate_layer: Type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.add_maxpool = add_maxpool
        
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio,
                rd_divisor,
                round_limit=0.0,
            )
        
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = gate_layer()

    def forward(self, x: Tensor) -> Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        
        return x * self.gate(x_se)


# =============================================================================
# Adaptive Perception Modules
# =============================================================================

class AdaptivePerceptionUnitLite(nn.Module):
    """
    Lightweight Adaptive Perception Unit (APU-Lite).
    
    A parameter-efficient version of APU using standard convolutions instead
    of deformable convolutions. Used when DCNv4 is disabled for the main stream.
    
    The APU-Lite preserves the dual-branch structure:
    - Top branch: Feature transformation with multi-branch convolution
    - Bottom branch: Skip connection to preserve high-resolution spatial details
    
    This design ensures high-resolution spatial details are preserved throughout
    the representation extraction process, preventing information loss during
    downsampling.
    
    Args:
        in_dim: Input dimension (C/2g where g is the number of groups).
        out_dim: Output dimension (fixed at 32 channels per group).
        kernel_size: Convolution kernel size for spatial feature extraction.
        mlp_ratio: MLP expansion ratio for channel mixing.
        act: Activation function (GELU for shallow layers, Swish for deep layers).
        
    Shape:
        - Input: (B, in_dim, H, W)
        - Output: (B, out_dim, H, W)
    """
    
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 32,
        kernel_size: int = 3,
        mlp_ratio: int = 4,
        act: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        self.act = act
        
        self.base_conv = nn.Conv2d(
            out_dim,
            out_dim,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            1,
            out_dim,
            bias=False,
        )
        self.base_norm = nn.BatchNorm2d(out_dim)

        self.add_conv = nn.Conv2d(out_dim, out_dim, 1, 1, 0, 1, out_dim, bias=False)
        self.add_norm = nn.BatchNorm2d(out_dim)
        self.mlp = MLP(out_dim, mlp_ratio, act=self.act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm(self.conv(x)))
        identity = x
        x = self.base_norm(self.base_conv(x)) + self.add_norm(self.add_conv(x)) + x
        x = self.mlp(x)
        return x + identity


class BypassStreamUnitLite(nn.Module):
    """
    Lightweight Bypass Stream Unit (f2 stream - Lite version).
    
    The bypass stream retains full-resolution spatial context across scales,
    integrating multi-level structural relationships critical for accurate
    lesion boundary definition (as described in Section III-B of the paper).
    
    Processing pipeline (Equation 1 in paper):
        f2'_i = pw(cb(pw(f2_i)))
    where pw denotes pointwise convolution with batch normalization,
    and cb denotes convolution block.
    
    This preserves crucial high-resolution features vital for capturing
    fine-grained details, particularly in complex cases where downsampling
    could result in loss of important spatial information.
    
    Args:
        in_dim: Input dimension (C/2).
        out_dim: Output dimension.
        
    Shape:
        - Input: (B, in_dim, H, W)
        - Output: (B, out_dim, H, W)
    """
    
    def __init__(self, in_dim: int = 32, out_dim: int = 32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_0 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv2d(
            in_dim, in_dim,
            padding=1,
            kernel_size=3,
            groups=in_dim,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class AdaptivePerceptionUnit(nn.Module):
    """
    Adaptive Perception Unit (APU) with Deformable Convolution.
    
    The APU is engineered to perform adaptive spatial sampling using DCNv4
    with learnable offsets. It excels in aggregating fine-grained edge
    representations while concurrently assimilating broad contextual information.
    
    Architecture (as described in Section III-B.1 of the paper):
    - Top branch: Deformable convolution dynamically adjusts receptive fields,
      enhancing the network's ability to capture fine-grained details such as
      polyp boundaries and structural variations. The output undergoes pointwise
      convolution with batch normalization and 3x3 convolution for refinement.
    - Bottom branch: Skip connection directly forwards the original feature
      representation to the fusion stage, preserving high-resolution spatial
      details and enhancing gradient flow during backpropagation.
    
    The deformable convolution incorporates learnable offsets for each sampling
    location (Equation 3 in paper):
        f''[p0] = Σ_k w[k] · f[p0 + k + s · Δpk]
    where s is the offset_scale hyperparameter controlling offset magnitude.
    
    Args:
        in_dim: Input dimension (H × W × C/2g).
        out_dim: Output dimension (fixed at 32 channels per group).
        kernel_size: Convolution kernel size.
        mlp_ratio: MLP expansion ratio.
        act: Activation function (GELU for shallow, Swish for deep layers).
        offset_scale: Scale factor (s) for deformable offsets Δpk.
                      Optimal value: 3.0 for APM (per ablation study Fig. 7).
        
    Shape:
        - Input: (B, in_dim, H, W) where in_dim = C/(2g)
        - Output: (B, out_dim, H, W)
    """
    
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 32,
        kernel_size: int = 3,
        mlp_ratio: int = 4,
        act: nn.Module = nn.GELU(),
        offset_scale: float = 2.5,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        
        # Deformable convolution for adaptive spatial sampling
        self.conv = DCNv4(
            in_dim,
            kernel_size=3,
            stride=1,
            pad=1,
            group=in_dim // out_dim,
            offset_scale=offset_scale,
            bias=False,
            layer_scale=1.0,
            dw_kernel_size=3,
            mlp_ratio=4.0,
            drop_path_rate=0.4,
            norm_layer='LN',
        )
        
        # Feature normalization
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        self.act = act
        
        # Re-parameterizable multi-branch convolution
        self.base_conv = nn.Conv2d(
            out_dim, out_dim,
            kernel_size, 1,
            (kernel_size - 1) // 2, 1,
            out_dim,
            bias=False,
        )
        self.base_norm = nn.BatchNorm2d(out_dim)

        self.add_conv = nn.Conv2d(out_dim, out_dim, 1, 1, 0, 1, out_dim, bias=False)
        self.add_norm = nn.BatchNorm2d(out_dim)
        
        # Channel-mixing MLP
        self.mlp = MLP(out_dim, mlp_ratio, act=self.act)

    def _process_chunk(self, x: Tensor) -> Tensor:
        """Process a single feature chunk."""
        x = self.act(self.norm(x))
        identity = x
        x = self.base_norm(self.base_conv(x)) + self.add_norm(self.add_conv(x)) + x
        x = self.mlp(x)
        return x + identity

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        # Apply deformable convolution
        x = x.reshape(B, H * W, C)
        x = self.conv(x)
        x = x.reshape(B, C, H, W)
        
        # Process in chunks for multi-scale feature extraction
        chunks = torch.chunk(x, C // self.out_dim, dim=1)
        processed = [self._process_chunk(chunk) for chunk in chunks]
        
        # Average across chunks
        return torch.stack(processed, dim=1).mean(dim=1)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """
        Fuse multi-branch convolution into a single convolution.
        
        Used during inference for computational efficiency.
        
        Returns:
            Fused convolutional layer.
        """
        # This would need proper implementation based on DCNv4's fuse method
        raise NotImplementedError(
            "Fusion requires DCNv4's fuse method. "
            "Implement based on your specific DCNv4 version."
        )


class BypassStreamUnit(nn.Module):
    """
    Bypass Stream Unit (f2 stream) with Deformable Convolution.
    
    The second stream (f2_i) in APM that retains full-resolution spatial context
    across scales, integrating multi-level structural relationships critical for
    accurate lesion boundary definition. Uses DCNv4 for enhanced feature extraction.
    
    This stream works in parallel with the APU stream, maintaining an equilibrium
    where localized micro-anatomical analysis and global contextual coherence
    are treated as interdependent requirements.
    
    Processing pipeline:
        f2'_i = pw(conv(pw(f2_i)))
    where conv uses DCNv4 for adaptive spatial sampling.
    
    Args:
        in_dim: Input dimension (C/2).
        out_dim: Output dimension.
        offset_scale: Scale factor (s) for deformable offsets.
                      Controls the magnitude of learnable offsets Δpk.
        
    Shape:
        - Input: (B, in_dim, H, W)
        - Output: (B, out_dim, H, W)
    """
    
    def __init__(
        self,
        in_dim: int = 32,
        out_dim: int = 32,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_0 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        
        self.conv = DCNv4(
            in_dim,
            kernel_size=3,
            stride=1,
            pad=1,
            group=in_dim // out_dim,
            offset_scale=offset_scale,
            bias=False,
        )
        
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        
        B, C, H, W = x.shape
        x = x.reshape(B, H * W, C)
        x = self.conv(x)
        x = x.reshape(B, C, H, W)

        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class AdaptivePerceptionModule(nn.Module):
    """
    Adaptive Perception Module (APM) for multi-scale feature processing.
    
    The APM is the core component of EAT, meticulously designed to accommodate
    and address the intricate morphological challenges posed by polyps. It
    employs a parallel two-stream framework as described in Section III-B
    of the paper.
    
    Architecture:
    - First stream (f1_i): Uses grouped APUs with adaptive receptive fields
      to dissect and analyze heterogeneous tissue regions, ensuring precise
      localization at sub-pixel scales. With C/2 input channels and g groups,
      each APU processes 32 channels independently (g = C/64).
    
    - Second stream (f2_i): Retains full-resolution spatial context across
      scales, integrating multi-level structural relationships critical for
      accurate lesion boundary definition.
    
    This co-dependent architecture maintains an equilibrium where localized
    micro-anatomical analysis and global contextual coherence are treated as
    interdependent requirements.
    
    Feature aggregation (Equation 2 in paper):
        f'_i = cb([f1'_i ; f2'_i])
    where [;] represents channel-wise concatenation.
    
    Args:
        in_dim: Input dimension (C from backbone feature map).
        out_dim: Output dimension (fixed at 32 for consistent feature size).
        kernel_size: Convolution kernel size for spatial feature extraction.
        mlp_ratio: MLP expansion ratio for channel mixing.
        shallow: Whether this is a shallow (early) layer.
                 True: uses GELU activation (for f2 features).
                 False: uses Swish activation (for f3, f4 features).
        gla_os: Offset scales (s) for [APU, Bypass] DCNv4 modules.
                Controls the magnitude of learnable offsets Δpk.
        
    Shape:
        - Input: (B, in_dim, H, W) where in_dim ∈ {C2, C3, C4}
        - Output: (B, out_dim, H, W)
    """
    
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 32,
        kernel_size: int = 3,
        mlp_ratio: int = 4,
        shallow: bool = True,
        gla_os: Tuple[float, float] = (2.5, 1.0),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        
        # Select activation based on layer depth
        self.act = nn.GELU() if shallow else Swish()
        
        # Adaptive Perception Unit
        apu_cls = AdaptivePerceptionUnit
        apu_kwargs = {
            'in_dim': in_dim // 2,
            'out_dim': out_dim,
            'kernel_size': kernel_size,
            'mlp_ratio': mlp_ratio,
            'act': self.act,
        }
        apu_kwargs['offset_scale'] = gla_os[0]
        self.apu = apu_cls(**apu_kwargs)
        
        # Bypass Stream Unit
        bypass_cls = BypassStreamUnitLite
        bypass_kwargs = {'in_dim': in_dim // 2, 'out_dim': out_dim}
        self.bypass = bypass_cls(**bypass_kwargs)

        # Feature fusion
        self.downsample = BasicConv2d(out_dim * 2, out_dim, 1, act=self.act)

    def forward(self, x: Tensor) -> Tensor:
        # Split input into two streams
        x_0, x_1 = x.chunk(2, dim=1)
        
        # Process through parallel streams
        x_0 = self.apu(x_0)
        x_1 = self.bypass(x_1)
        
        # Concatenate and fuse
        x = torch.cat([x_0, x_1], dim=1)
        x = self.downsample(x)
        
        return x


# =============================================================================
# Comprehensive Feature Extractor
# =============================================================================

class CFEAttention(nn.Module):
    """
    Comprehensive Feature Extractor (CFE) - Attention Component.
    
    Implements the self-attention mechanism of CFE as described in Section III-C
    of the paper. The attention selectively attends to relevant local representations
    to capture broader contextual information.
    
    The process (Equation 5 in paper):
        Q = Wq·ψ(f'12), K = Wk·ψ(f'12), V = Wv·ψ(f'12)
        fa = softmax(QK^T / √dk) · V
    
    where ψ denotes average pooling with stride r for downsampling.
    
    Args:
        channels: Number of input channels.
        r: Spatial reduction factor (stride for average pooling).
        heads: Number of attention heads for multi-head attention.
        
    Shape:
        - Input: (B, channels, H, W)
        - Output: (B, channels, H//r, W//r)
    """
    
    def __init__(self, channels: int, r: int, heads: int) -> None:
        super().__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split(
            [self.head_dim, self.head_dim, self.head_dim],
            dim=2,
        )
        
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        
        return x


class CFEUpsample(nn.Module):
    """
    Comprehensive Feature Extractor (CFE) - Upsampling Component.
    
    Restores spatial resolution and refines local details as described in
    Section III-C of the paper (Equation 6):
        f12 = pc(τ(fa))
    
    where τ(·) is the transposed convolution operation used to restore
    the local spatial representation, and pc(·) is the pointwise convolution
    operation responsible for enhancing the local representation details.
    
    This two-step refinement process ensures both global contextual information
    and fine local structures are preserved, leading to improved segmentation
    performance.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        r: Upsampling factor (should match the downsampling factor in CFEAttention).
        
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H*r, W*r)
    """
    
    def __init__(self, in_channels: int, out_channels: int, r: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv_trans = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=r,
            stride=r,
            groups=in_channels,
        )
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class ComprehensiveFeatureExtractor(nn.Module):
    """
    Comprehensive Feature Extractor (CFE).
    
    The CFE employs self-attention mechanisms to refine local feature maps
    into intermediate feature maps containing both fine-grained local details
    and global contextual information. This module is the core component for
    integrating multi-scale feature representations in the EAT architecture.
    
    As described in Section III-C of the paper, CFE consists of two main
    components:
    
    1. **CFEAttention**: Applies self-attention using Equation 5:
       ``f'_12 = Attention(Q, K, V) · W^O``
       where Q, K, V are obtained via convolutional projections of the
       concatenated features f'_1 and f'_2.
    
    2. **CFEUpsample**: Refines the attention output using Equation 6:
       ``f_12 = Upsample(Norm(DepthWise(f'_12 · σ(f'_12))))``
       incorporating gating mechanisms and depthwise separable convolutions.
    
    The CFE processes the concatenated shallow features (f'_1) processed
    through deformable convolution and APM-processed f'_2, producing refined
    feature maps that aggregate both detail-preserving and semantically-rich
    information at the target resolution.
    
    Reference:
        IEEE TMI Paper, Section III-C: "Comprehensive Feature Extractor"
    
    Args:
        in_channels: Number of input channels (typically out_dim * 2 from
            concatenated features).
        out_channels: Number of output channels (segmentation classes).
        r: Spatial reduction/expansion factor for attention and upsampling.
            Default: 4.
        heads: Number of attention heads for multi-head self-attention.
            Default: 2.
        
    Shape:
        - Input: (B, in_channels, H, W) - concatenated feature maps
        - Output: (B, out_channels, H', W') - refined and upsampled features
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r: int = 4,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.attn = CFEAttention(in_channels, r=r, heads=heads)
        self.upsample = CFEUpsample(
            in_channels=in_channels,
            out_channels=out_channels,
            r=r,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(self.attn(x))


# =============================================================================
# Main EAT Model
# =============================================================================

class EAT(nn.Module):
    """
    Endoscopic Adaptive Transformer (EAT) for Polyp Segmentation.
    
    EAT is a framework specifically designed for polyp segmentation in
    endoscopic images. It addresses the unique challenges of endoscopic
    imaging, including non-uniform illumination, motion blur, and the
    irregular morphology of polyps, through an adaptive perceptive-field
    mechanism that dynamically captures both fine-grained local details
    and extensive contextual information.
    
    Published in IEEE Transactions on Medical Imaging (TMI).
    DOI: 10.1109/TMI.2025.3615677
    
    Architecture Overview (Section III-A):
    ------------------------------------
    Given an input image I ∈ R^{H×W×C}, EAT operates as follows:
    
    1. **PVT-v2 Backbone**: Extracts hierarchical multi-scale feature
       representations {f_1, f_2, f_3, f_4} with progressive spatial
       reduction and channel expansion.
    
    2. **Shallow Feature Processing**: The shallowest features f_1 are
       enhanced using deformable convolution to improve edge detection
       and preserve fine spatial details. The optimal offset scale for
       this stage is 1.0 (Table VI).
    
    3. **Adaptive Perception Module (APM)**: Deeper features {f_2, f_3, f_4}
       are processed through APMs, which employ parallel two-stream
       architectures:
       - APU Stream: Uses Adaptive Perception Units with deformable
         convolutions (optimal offset scale 3.0) for adaptive receptive fields
       - Bypass Stream: Preserves full-resolution spatial context
    
    4. **Comprehensive Feature Extractor (CFE)**: Concatenated features
       f'_1 and f'_2 are refined through self-attention mechanisms to
       produce f_12, capturing both local details and global context.
    
    5. **Feature Aggregation**: High-level features f'_3 and f'_4 are
       combined via upsampling and concatenation to produce f_34, which
       is then fused with f_12 for final prediction.
    
    Key Components:
    - APM with DCNv4: Enables adaptive receptive field adjustment
    - CFE with Self-Attention: Integrates multi-scale feature information
    - Learnable Offset Scales: Control the magnitude of spatial sampling offsets
    
    Reference:
        "EAT: Endoscopic Adaptive Transformer for Polyp Segmentation"
        IEEE Transactions on Medical Imaging, 2025
    
    Args:
        in_channels: Number of input image channels. Default: 3 (RGB).
        out_channels: Number of output segmentation classes. Default: 1.
        dims: Feature dimensions at each backbone stage {f_1, f_2, f_3, f_4}.
            Default: (64, 128, 320, 512).
        out_dim: Unified output dimension for intermediate feature processing.
            Default: 32.
        kernel_size: Kernel size for convolution operations. Default: 3.
        mlp_ratio: MLP expansion ratio in APU/Bypass modules. Default: 4.
        gla_os: Tuple of offset scales for (APU, Bypass) DCNv4 modules.
            Controls the magnitude of learnable offsets Δp_k.
            Default: (2.5, 1.0). Paper optimal: (3.0, 1.0).
        L_feature_v4: Whether to use DCNv4 for shallow feature (f_1) processing.
            Default: True.
        L_feature_os: Offset scale for shallow feature DCNv4.
            Default: 2.0. Paper optimal: 1.0 (Table VI).
        model_dir: Path to pretrained PVT-v2 backbone weights.
            Default: './pretrained/pvt_v2_b2.pth'.
        
    Shape:
        - Input: (B, in_channels, H, W) - Endoscopic images
        - Output: (B, out_channels, H, W) - Segmentation masks
        
    Example:
        >>> model = EAT(in_channels=3, out_channels=1)
        >>> x = torch.randn(1, 3, 352, 352)
        >>> output = model(x)
        >>> print(output.shape)  # (1, 1, 352, 352)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        dims: Sequence[int] = (64, 128, 320, 512),
        out_dim: int = 32,
        kernel_size: int = 3,
        mlp_ratio: int = 4,
        gla_os: Tuple[float, float] = (2.5, 1.0),
        L_feature_v4: bool = True,
        L_feature_os: float = 2.0,
        model_dir: Optional[str] = './pretrained/pvt_v2_b2.pth',
    ) -> None:
        super().__init__()
        
        # Validate inputs
        if len(dims) != 4:
            raise ValueError(f"dims must have exactly 4 stages, got {len(dims)}")
        
        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = tuple(dims)
        self.out_dim = out_dim
        
        # Initialize backbone
        self.backbone = pvt_v2_b2(in_chans=in_channels, embed_dims=list(dims))
        self._load_pretrained_weights(model_dir)

        # Unpack feature dimensions
        c1_in, c2_in, c3_in, c4_in = dims

        # Adaptive Perception Modules for multi-scale processing
        self.apm2 = AdaptivePerceptionModule(
            in_dim=c2_in,
            out_dim=out_dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            shallow=True,
            gla_os=gla_os,
        )
        self.apm3 = AdaptivePerceptionModule(
            in_dim=c3_in,
            out_dim=out_dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            shallow=False,
            gla_os=gla_os,
        )
        self.apm4 = AdaptivePerceptionModule(
            in_dim=c4_in,
            out_dim=out_dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            shallow=False,
            gla_os=gla_os,
        )

        # Feature fusion for high-level features
        self.fuse2 = nn.Sequential(
            BasicConv2d(out_dim * 2, out_dim, 1, 1),
            nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False),
        )

        # Low-level feature processing
        if L_feature_v4:
            self.dfconv = DFConv(c1_in, out_dim, 3, 1, 1, offset_scale=L_feature_os)
        else:
            self.dfconv = BasicConv2d(c1_in, out_dim, 3, 1, 1)

        # Comprehensive Feature Extractor
        self.cfe = ComprehensiveFeatureExtractor(
            in_channels=out_dim * 2,
            out_channels=out_channels,
            r=4,
            heads=2,
        )
        
        # Final feature fusion
        self.fuse = BasicConv2d(out_dim, out_dim, 1)
        
        logger.info(
            f"EAT model initialized with {self.count_parameters() / 1e6:.2f}M parameters"
        )

    def _load_pretrained_weights(self, model_dir: Optional[str]) -> None:
        """
        Load pretrained backbone weights.
        
        Args:
            model_dir: Path to pretrained weights file.
        """
        if model_dir is None:
            logger.info("No pretrained weights specified, training from scratch")
            return
            
        model_path = Path(model_dir)
        if not model_path.is_file():
            logger.warning(
                f"Pretrained weights not found at {model_dir!r}. "
                f"Continuing without loading pretrained weights."
            )
            return
        
        try:
            checkpoint = torch.load(model_dir, map_location='cpu')
            model_dict = self.backbone.state_dict()
            
            # Filter and load matching weights
            pretrained_dict = {
                k: v for k, v in checkpoint.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
            
            logger.info(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} "
                f"pretrained weights from {model_dir}"
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: Whether to count only trainable parameters.
            
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _upsample(
        x: Tensor,
        size: Tuple[int, int],
        align_corners: bool = False,
    ) -> Tensor:
        """
        Bilinear upsampling wrapper.
        
        Args:
            x: Input tensor.
            size: Target (H, W) size.
            align_corners: Whether to align corners during interpolation.
            
        Returns:
            Upsampled tensor.
        """
        return F.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through EAT model.
        
        Args:
            x: Input image tensor of shape (B, C, H, W).
            
        Returns:
            Segmentation output of shape (B, out_channels, H, W).
        """
        # Extract multi-scale features from backbone
        c1, c2, c3, c4 = self.backbone(x)
        
        # Process high-level features through APMs
        _c4 = self.apm4(c4)
        _c4 = self._upsample(_c4, c3.shape[2:])
        
        _c3 = self.apm3(c3)
        _c2 = self.apm2(c2)

        # Fuse high-level features
        high_level_fused = torch.cat([
            self._upsample(_c4, c2.shape[2:]),
            self._upsample(_c3, c2.shape[2:]),
        ], dim=1)
        output = self.fuse2(high_level_fused)

        # Process low-level features
        L_feature = self.dfconv(c1)
        
        # Process and upsample mid-level features
        H_feature = self.fuse(_c2)
        H_feature = self._upsample(H_feature, L_feature.shape[2:])

        # Comprehensive feature extraction
        output2 = self.cfe(torch.cat([H_feature, L_feature], dim=1))

        # Final upsampling to original resolution
        output = F.interpolate(output, scale_factor=8, mode='bilinear')
        output2 = F.interpolate(output2, scale_factor=4, mode='bilinear')

        return output + output2

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'EAT':
        """
        Create EAT model from configuration object.
        
        Args:
            config: ModelConfig instance.
            
        Returns:
            Initialized EAT model.
        """
        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            dims=config.dims,
            out_dim=config.out_dim,
            kernel_size=config.kernel_size,
            mlp_ratio=config.mlp_ratio,
            gla_os=config.gla_os,
            L_feature_v4=config.l_feature_v4,
            L_feature_os=config.l_feature_os,
            model_dir=config.pretrained_path,
        )


# =============================================================================
# Model Registry and Factory
# =============================================================================

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'eat': EAT,
    'eat_base': EAT,
}


def register_eat_model(name: str) -> Callable:
    """
    Decorator to register a model in the registry.
    
    Args:
        name: Name to register the model under.
        
    Returns:
        Decorator function.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(
    model_name: str,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a model from the registry.
    
    Args:
        model_name: Name of the model to build.
        **kwargs: Model configuration arguments.
        
    Returns:
        Instantiated model.
        
    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available}"
        )
    
    return _MODEL_REGISTRY[model_name](**kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 352, 352),
) -> Dict[str, Any]:
    """
    Compute model complexity metrics.
    
    Args:
        model: PyTorch model.
        input_size: Input tensor size (B, C, H, W).
        
    Returns:
        Dictionary with complexity metrics.
    """
    from torch.profiler import profile, ProfilerActivity
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate FLOPs using profiler
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        _ = model(dummy_input)
    
    flops = sum(event.flops for event in prof.key_averages() if event.flops is not None)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6,
        'flops': flops,
        'gflops': flops / 1e9,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    """
    Test script for EAT model.
    
    Demonstrates model instantiation, forward pass, and basic testing.
    """
    import time
    
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    input_size = (batch_size, 3, 352, 352)
    
    print(f"Testing EAT model on device: {device}")
    print(f"Input size: {input_size}")
    print("-" * 50)
    
    # Create model
    model = EAT(in_channels=3, out_channels=2).to(device)
    model.eval()
    
    # Print model info
    print(f"Total parameters: {model.count_parameters(trainable_only=False) / 1e6:.2f}M")
    print(f"Trainable parameters: {model.count_parameters(trainable_only=True) / 1e6:.2f}M")
    print("-" * 50)
    
    # Test forward pass
    x = torch.randn(input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    num_iterations = 10
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed_time = (time.time() - start_time) / num_iterations * 1000
    
    print(f"Output shape: {output.shape}")
    print(f"Average inference time: {elapsed_time:.2f} ms")
    print(f"Throughput: {1000 / elapsed_time:.2f} FPS")
    print("-" * 50)
    
    # Test with config
    print("Testing model creation from config...")
    config = ModelConfig(
        in_channels=3,
        out_channels=1,
        dims=(64, 128, 320, 512),
        out_dim=32,
    )
    model_from_config = EAT.from_config(config).to(device)
    print(f"Model from config created successfully!")
    print(f"Parameters: {model_from_config.count_parameters() / 1e6:.2f}M")
