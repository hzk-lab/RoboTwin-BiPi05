"""
TwinVLA1 - Dual VLA with Joint Attention

双VLA架构实现，使用独立的左右臂VLA并通过causal joint attention进行信息交流。

核心组件:
- noise_utils: 左右臂噪声添加和动作提取工具
- joint_attention: Causal joint attention mask
- dual_vla: 双VLA核心模型
"""

# 延迟导入以避免循环依赖
__all__ = [
    # noise_utils
    "add_noise_left",
    "add_noise_right",
    "extract_left_action",
    "extract_right_action",
    "prepare_left_actions",
    "prepare_right_actions",
    "merge_actions",
    "ACTION_DIM",
    "LEFT_ARM_DIM",
    "RIGHT_ARM_DIM",
    # joint_attention
    "make_dual_vla_ar_mask",
    "make_cross_vla_attn_mask",
    "make_suffix_cross_attn_mask",
    # dual_vla
    "DualVLAConfig",
    "DualVLA",
    "DualVLAPolicy",
    "create_dual_vla_policy",
]


def __getattr__(name):
    """延迟导入"""
    if name in (
        "add_noise_left",
        "add_noise_right",
        "extract_left_action",
        "extract_right_action",
        "prepare_left_actions",
        "prepare_right_actions",
        "merge_actions",
        "ACTION_DIM",
        "LEFT_ARM_DIM",
        "RIGHT_ARM_DIM",
    ):
        from . import noise_utils
        return getattr(noise_utils, name)
    elif name in (
        "make_dual_vla_ar_mask",
        "make_cross_vla_attn_mask",
        "make_suffix_cross_attn_mask",
    ):
        from . import joint_attention
        return getattr(joint_attention, name)
    elif name in (
        "DualVLAConfig",
        "DualVLA",
        "DualVLAPolicy",
        "create_dual_vla_policy",
    ):
        from . import dual_vla
        return getattr(dual_vla, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

