"""
EAT: Endoscopic Adaptive Transformer for Polyp Segmentation.

This package provides the EAT model implementation for polyp segmentation in
endoscopic images. EAT addresses the unique challenges of endoscopic imaging—
non-uniform illumination, motion blur, and irregular polyp morphology—through
an adaptive perceptive-field mechanism that dynamically captures both
fine-grained local details and extensive contextual information.

Key Components:
    - **PVT-v2 Backbone**: Hierarchical multi-scale feature extraction
    - **Adaptive Perception Module (APM)**: Parallel two-stream processing
      with Adaptive Perception Units (APU) using deformable convolutions
    - **Comprehensive Feature Extractor (CFE)**: Self-attention based
      feature refinement and aggregation

Published in IEEE Transactions on Medical Imaging (TMI).
DOI: 10.1109/TMI.2025.3615677

Reference:
    "EAT: Endoscopic Adaptive Transformer for Polyp Segmentation"

Example:
    >>> from eat_model import EAT, ModelConfig, build_model
    >>> 
    >>> # Method 1: Direct instantiation
    >>> model = EAT(in_channels=3, out_channels=1)
    >>> 
    >>> # Method 2: Using configuration
    >>> config = ModelConfig(in_channels=3, out_channels=2)
    >>> model = EAT.from_config(config)
    >>> 
    >>> # Method 3: Using factory function
    >>> model = build_model('eat', in_channels=3, out_channels=1)
"""

from .eat import (
    # Main model
    EAT,
    
    # Configuration
    ModelConfig,
    PVTConfig,
    
    # Factory functions
    build_model,
    register_eat_model,
    
    # Backbone
    PyramidVisionTransformerImpr,
    pvt_v2_b2,
    
    # Building blocks
    Block,
    Attention,
    FlashAttention,
    Mlp,
    OverlapPatchEmbed,
    
    # Adaptive perception modules
    AdaptivePerceptionModule,
    AdaptivePerceptionUnit,
    AdaptivePerceptionUnitLite,
    BypassStreamUnit,
    BypassStreamUnitLite,
    
    # Feature extractors
    ComprehensiveFeatureExtractor,
    SEModule,
    
    # Utilities
    get_model_complexity,
)

__version__ = '1.0.0'
__author__ = 'EAT Authors'

__all__ = [
    # Main model
    'EAT',
    
    # Configuration
    'ModelConfig',
    'PVTConfig',
    
    # Factory functions
    'build_model',
    'register_eat_model',
    
    # Backbone
    'PyramidVisionTransformerImpr',
    'pvt_v2_b2',
    
    # Building blocks
    'Block',
    'Attention',
    'FlashAttention',
    'Mlp',
    'OverlapPatchEmbed',
    
    # Adaptive perception modules
    'AdaptivePerceptionModule',
    'AdaptivePerceptionUnit',
    'AdaptivePerceptionUnitLite',
    'BypassStreamUnit',
    'BypassStreamUnitLite',
    
    # Feature extractors
    'ComprehensiveFeatureExtractor',
    'SEModule',
    
    # Utilities
    'get_model_complexity',
]
