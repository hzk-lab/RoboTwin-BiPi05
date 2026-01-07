"""
Noise Utilities for TwinVLA1 Dual VLA

实现左右臂VLA的噪声添加和动作提取逻辑。

噪声添加策略:
- 左臂VLA: [7维左臂数据 + noise, 25维纯noise]
- 右臂VLA: [7维纯noise, 7维右臂数据 + noise, 18维纯noise]

有效维度:
- 左臂: 0-6 (7维)
- 右臂: 7-13 (7维)
"""

import jax
import jax.numpy as jnp

# 维度常量
ACTION_DIM = 32
LEFT_ARM_START = 0
LEFT_ARM_END = 7  # exclusive
RIGHT_ARM_START = 7
RIGHT_ARM_END = 14  # exclusive

LEFT_ARM_DIM = LEFT_ARM_END - LEFT_ARM_START  # 7
RIGHT_ARM_DIM = RIGHT_ARM_END - RIGHT_ARM_START  # 7


def prepare_left_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """
    准备左臂VLA的目标动作。
    
    将原始14维动作转换为32维格式，只保留左臂数据:
    - 前7维: 左臂数据
    - 后25维: 零 (训练时会被纯噪声替代)
    
    Args:
        actions: [*batch, action_horizon, 14] 原始双臂动作
        
    Returns:
        left_actions: [*batch, action_horizon, 32] 左臂目标动作
    """
    batch_shape = actions.shape[:-1]
    
    # 提取左臂数据 (前7维)
    left_arm = actions[..., LEFT_ARM_START:LEFT_ARM_END]  # [*b, ah, 7]
    
    # 零填充到32维
    padding_shape = (*batch_shape, ACTION_DIM - LEFT_ARM_DIM)
    zeros_padding = jnp.zeros(padding_shape, dtype=actions.dtype)
    
    # 拼接: [left_7d, zeros_25d]
    left_actions = jnp.concatenate([left_arm, zeros_padding], axis=-1)
    
    return left_actions


def prepare_right_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """
    准备右臂VLA的目标动作。
    
    将原始14维动作转换为32维格式，只保留右臂数据:
    - 前7维: 零 (训练时会被纯噪声替代)
    - 7-13维: 右臂数据
    - 后18维: 零 (训练时会被纯噪声替代)
    
    Args:
        actions: [*batch, action_horizon, 14] 原始双臂动作
        
    Returns:
        right_actions: [*batch, action_horizon, 32] 右臂目标动作
    """
    batch_shape = actions.shape[:-1]
    
    # 提取右臂数据 (7-13维)
    right_arm = actions[..., RIGHT_ARM_START:RIGHT_ARM_END]  # [*b, ah, 7]
    
    # 前7维零填充
    prefix_shape = (*batch_shape, LEFT_ARM_DIM)
    zeros_prefix = jnp.zeros(prefix_shape, dtype=actions.dtype)
    
    # 后18维零填充
    suffix_shape = (*batch_shape, ACTION_DIM - RIGHT_ARM_END)
    zeros_suffix = jnp.zeros(suffix_shape, dtype=actions.dtype)
    
    # 拼接: [zeros_7d, right_7d, zeros_18d]
    right_actions = jnp.concatenate([zeros_prefix, right_arm, zeros_suffix], axis=-1)
    
    return right_actions


def add_noise_left(
    rng,
    actions: jnp.ndarray,
    time: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    为左臂VLA添加噪声。
    
    加噪方式: [7维左臂数据 + noise, 25维纯noise]
    - 前7维: flow matching 插值 (x_t = t * noise + (1-t) * actions)
    - 后25维: 纯高斯噪声
    
    Args:
        rng: JAX random key
        actions: [*batch, action_horizon, 14] 原始双臂动作
        time: [*batch] 时间步 t ∈ (0, 1)
        
    Returns:
        x_t: [*batch, action_horizon, 32] 加噪后的动作
        u_t: [*batch, action_horizon, 32] 速度场目标 (noise - actions)
    """
    batch_shape = actions.shape[:-2]
    action_horizon = actions.shape[-2]
    
    # 准备左臂目标动作
    left_actions = prepare_left_actions(actions)  # [*b, ah, 32]
    
    # 生成完整32维噪声
    noise_rng, pure_noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (*batch_shape, action_horizon, ACTION_DIM))
    
    # 扩展时间维度
    time_expanded = time[..., None, None]  # [*b, 1, 1]
    
    # 前7维: flow matching 插值
    left_arm_data = left_actions[..., :LEFT_ARM_DIM]  # [*b, ah, 7]
    left_arm_noise = noise[..., :LEFT_ARM_DIM]  # [*b, ah, 7]
    x_t_left = time_expanded * left_arm_noise + (1 - time_expanded) * left_arm_data
    
    # 后25维: 纯噪声 (不依赖 time)
    pure_noise = jax.random.normal(pure_noise_rng, (*batch_shape, action_horizon, ACTION_DIM - LEFT_ARM_DIM))
    
    # 拼接加噪结果
    x_t = jnp.concatenate([x_t_left, pure_noise], axis=-1)
    
    # 计算速度场目标
    # 前7维: noise - actions (标准 flow matching)
    # 后25维: 0 (纯噪声无目标)
    u_t_left = left_arm_noise - left_arm_data
    u_t_padding = jnp.zeros((*batch_shape, action_horizon, ACTION_DIM - LEFT_ARM_DIM), dtype=actions.dtype)
    u_t = jnp.concatenate([u_t_left, u_t_padding], axis=-1)
    
    return x_t, u_t


def add_noise_right(
    rng,
    actions: jnp.ndarray,
    time: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    为右臂VLA添加噪声。
    
    加噪方式: [7维纯noise, 7维右臂数据 + noise, 18维纯noise]
    - 前7维: 纯高斯噪声
    - 7-13维: flow matching 插值
    - 后18维: 纯高斯噪声
    
    Args:
        rng: JAX random key
        actions: [*batch, action_horizon, 14] 原始双臂动作
        time: [*batch] 时间步 t ∈ (0, 1)
        
    Returns:
        x_t: [*batch, action_horizon, 32] 加噪后的动作
        u_t: [*batch, action_horizon, 32] 速度场目标
    """
    batch_shape = actions.shape[:-2]
    action_horizon = actions.shape[-2]
    
    # 准备右臂目标动作
    right_actions = prepare_right_actions(actions)  # [*b, ah, 32]
    
    # 生成各部分噪声
    prefix_rng, mid_rng, suffix_rng = jax.random.split(rng, 3)
    
    # 扩展时间维度
    time_expanded = time[..., None, None]  # [*b, 1, 1]
    
    # 前7维: 纯噪声
    prefix_noise = jax.random.normal(prefix_rng, (*batch_shape, action_horizon, LEFT_ARM_DIM))
    
    # 7-13维: flow matching 插值
    right_arm_data = right_actions[..., RIGHT_ARM_START:RIGHT_ARM_END]  # [*b, ah, 7]
    right_arm_noise = jax.random.normal(mid_rng, (*batch_shape, action_horizon, RIGHT_ARM_DIM))
    x_t_right = time_expanded * right_arm_noise + (1 - time_expanded) * right_arm_data
    
    # 后18维: 纯噪声
    suffix_noise = jax.random.normal(suffix_rng, (*batch_shape, action_horizon, ACTION_DIM - RIGHT_ARM_END))
    
    # 拼接加噪结果
    x_t = jnp.concatenate([prefix_noise, x_t_right, suffix_noise], axis=-1)
    
    # 计算速度场目标
    # 前7维: 0 (纯噪声无目标)
    # 7-13维: noise - actions (标准 flow matching)
    # 后18维: 0 (纯噪声无目标)
    u_t_prefix = jnp.zeros((*batch_shape, action_horizon, LEFT_ARM_DIM), dtype=actions.dtype)
    u_t_right = right_arm_noise - right_arm_data
    u_t_suffix = jnp.zeros((*batch_shape, action_horizon, ACTION_DIM - RIGHT_ARM_END), dtype=actions.dtype)
    u_t = jnp.concatenate([u_t_prefix, u_t_right, u_t_suffix], axis=-1)
    
    return x_t, u_t


def extract_left_action(actions: jnp.ndarray) -> jnp.ndarray:
    """
    从32维输出中提取左臂动作 (前7维)。
    
    Args:
        actions: [*batch, action_horizon, 32] 左臂VLA输出
        
    Returns:
        left_actions: [*batch, action_horizon, 7] 左臂动作
    """
    return actions[..., LEFT_ARM_START:LEFT_ARM_END]


def extract_right_action(actions: jnp.ndarray) -> jnp.ndarray:
    """
    从32维输出中提取右臂动作 (7-13维)。
    
    Args:
        actions: [*batch, action_horizon, 32] 右臂VLA输出
        
    Returns:
        right_actions: [*batch, action_horizon, 7] 右臂动作
    """
    return actions[..., RIGHT_ARM_START:RIGHT_ARM_END]


def merge_actions(
    left_actions: jnp.ndarray,
    right_actions: jnp.ndarray,
) -> jnp.ndarray:
    """
    合并左右臂动作为14维输出。
    
    Args:
        left_actions: [*batch, action_horizon, 7] 左臂动作
        right_actions: [*batch, action_horizon, 7] 右臂动作
        
    Returns:
        actions: [*batch, action_horizon, 14] 合并后的双臂动作
    """
    return jnp.concatenate([left_actions, right_actions], axis=-1)


def create_initial_noise_left(
    rng,
    batch_size: int,
    action_horizon: int,
) -> jnp.ndarray:
    """
    创建左臂VLA的初始噪声。
    
    格式: [random_7d, random_25d] - 全部随机，去噪后只取前7维
    """
    return jax.random.normal(rng, (batch_size, action_horizon, ACTION_DIM))


def create_initial_noise_right(
    rng,
    batch_size: int,
    action_horizon: int,
) -> jnp.ndarray:
    """
    创建右臂VLA的初始噪声。
    
    格式: [random_7d, random_7d, random_18d] - 全部随机，去噪后只取7-13维
    """
    return jax.random.normal(rng, (batch_size, action_horizon, ACTION_DIM))
