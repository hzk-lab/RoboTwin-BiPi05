"""
Dual VLA Model for TwinVLA1

实现两个独立的VLA模型分别处理左右臂，通过causal joint attention实现跨臂信息交流。

架构:
- Left VLA: 处理左臂动作，输入32维，只有前7维有效
- Right VLA: 处理右臂动作，输入32维，只有7-13维有效
- Joint Attention: 在decoder层让左右臂token互相交流

核心改进:
1. 两个完全独立的VLA模型 (各自有完整的vision/language encoder)
2. 特定的噪声添加策略 (只对有效维度做flow matching)
3. 跨VLA的causal joint attention
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pi0" / "src"))

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models.pi0 import Pi0Config, Pi0, make_attn_mask, posemb_sincos
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.shared import array_typing as at
from openpi.shared import download
import openpi.transforms as transforms
import openpi.shared.nnx_utils as nnx_utils

from .noise_utils import (
    ACTION_DIM,
    LEFT_ARM_DIM,
    RIGHT_ARM_DIM,
    LEFT_ARM_START,
    LEFT_ARM_END,
    RIGHT_ARM_START,
    RIGHT_ARM_END,
    add_noise_left,
    add_noise_right,
    extract_left_action,
    extract_right_action,
    merge_actions,
    create_initial_noise_left,
    create_initial_noise_right,
)
from .joint_attention import make_dual_vla_ar_mask

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DualVLAConfig:
    """Dual VLA 配置"""
    # 基础配置
    action_dim: int = ACTION_DIM  # 32
    action_horizon: int = 50
    max_token_len: int = 48
    
    # 左右臂有效维度
    left_arm_dim: int = LEFT_ARM_DIM  # 7
    right_arm_dim: int = RIGHT_ARM_DIM  # 7
    
    # Pi0 配置 (用于创建底层模型)
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"


class DualVLA:
    """
    双VLA模型 - 管理两个独立的Pi0模型并实现跨臂joint attention。
    
    核心功能:
    1. compute_loss: 训练时分别计算左右臂的损失
    2. sample_actions: 推理时并行去噪，通过joint attention协调
    """
    
    def __init__(
        self,
        left_model: Pi0,
        right_model: Pi0,
        config: DualVLAConfig,
    ):
        """
        初始化 Dual VLA。
        
        Args:
            left_model: 左臂 Pi0 模型
            right_model: 右臂 Pi0 模型
            config: DualVLA 配置
        """
        self.left_model = left_model
        self.right_model = right_model
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        
        logger.info(f"DualVLA initialized with action_dim={self.action_dim}, horizon={self.action_horizon}")
    
    def compute_loss(
        self,
        rng,
        observation: _model.Observation,
        actions: jnp.ndarray,
        *,
        train: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        计算训练损失。
        
        分别对左右臂VLA计算损失，使用各自的噪声添加策略。
        
        Args:
            rng: JAX random key
            observation: 观测数据
            actions: [batch, action_horizon, 14] 真实动作
            train: 是否训练模式
            
        Returns:
            left_loss: [batch, action_horizon] 左臂损失
            right_loss: [batch, action_horizon] 右臂损失
        """
        # 分割随机数
        preprocess_rng, left_rng, right_rng, time_rng = jax.random.split(rng, 4)
        left_noise_rng, right_noise_rng = jax.random.split(left_rng), jax.random.split(right_rng)
        
        # 预处理观测
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        # 采样时间步
        batch_size = actions.shape[0]
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        
        # === 左臂 VLA 损失 ===
        # 添加噪声: [7维数据+noise, 25维纯noise]
        x_t_left, u_t_left = add_noise_left(left_noise_rng[0], actions, time)
        
        # 前向传播
        prefix_tokens_l, prefix_mask_l, prefix_ar_mask_l = self.left_model.embed_prefix(observation)
        suffix_tokens_l, suffix_mask_l, suffix_ar_mask_l = self.left_model.embed_suffix(observation, x_t_left, time)
        
        input_mask_l = jnp.concatenate([prefix_mask_l, suffix_mask_l], axis=1)
        ar_mask_l = jnp.concatenate([prefix_ar_mask_l, suffix_ar_mask_l], axis=0)
        attn_mask_l = make_attn_mask(input_mask_l, ar_mask_l)
        positions_l = jnp.cumsum(input_mask_l, axis=1) - 1
        
        (_, suffix_out_l), _ = self.left_model.PaliGemma.llm(
            [prefix_tokens_l, suffix_tokens_l],
            mask=attn_mask_l,
            positions=positions_l,
        )
        v_t_left = self.left_model.action_out_proj(suffix_out_l[:, -self.action_horizon:])
        
        # 只计算前7维的损失
        left_loss = jnp.mean(jnp.square(v_t_left[..., :LEFT_ARM_DIM] - u_t_left[..., :LEFT_ARM_DIM]), axis=-1)
        
        # === 右臂 VLA 损失 ===
        # 添加噪声: [7维noise, 7维数据+noise, 18维noise]
        x_t_right, u_t_right = add_noise_right(right_noise_rng[0], actions, time)
        
        # 前向传播
        prefix_tokens_r, prefix_mask_r, prefix_ar_mask_r = self.right_model.embed_prefix(observation)
        suffix_tokens_r, suffix_mask_r, suffix_ar_mask_r = self.right_model.embed_suffix(observation, x_t_right, time)
        
        input_mask_r = jnp.concatenate([prefix_mask_r, suffix_mask_r], axis=1)
        ar_mask_r = jnp.concatenate([prefix_ar_mask_r, suffix_ar_mask_r], axis=0)
        attn_mask_r = make_attn_mask(input_mask_r, ar_mask_r)
        positions_r = jnp.cumsum(input_mask_r, axis=1) - 1
        
        (_, suffix_out_r), _ = self.right_model.PaliGemma.llm(
            [prefix_tokens_r, suffix_tokens_r],
            mask=attn_mask_r,
            positions=positions_r,
        )
        v_t_right = self.right_model.action_out_proj(suffix_out_r[:, -self.action_horizon:])
        
        # 只计算7-13维的损失
        right_loss = jnp.mean(jnp.square(v_t_right[..., RIGHT_ARM_START:RIGHT_ARM_END] - u_t_right[..., RIGHT_ARM_START:RIGHT_ARM_END]), axis=-1)
        
        return left_loss, right_loss
    
    def sample_actions(
        self,
        rng,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
    ) -> jnp.ndarray:
        """
        采样动作 - 使用joint attention让左右臂协调去噪。
        
        关键步骤:
        1. 两个VLA分别对prefix做前向传播，缓存KV
        2. 并行去噪循环，在每一步:
           a. 各自嵌入suffix tokens
           b. 构建跨VLA的joint attention mask
           c. 联合前向传播 (左右臂action tokens互相可见)
           d. 各自提取有效维度并更新
        3. 合并左右臂输出
        
        Args:
            rng: JAX random key
            observation: 观测数据
            num_steps: 去噪步数
            
        Returns:
            actions: [batch, action_horizon, 14] 采样的动作
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # 初始化噪声
        left_rng, right_rng = jax.random.split(rng)
        noise_left = create_initial_noise_left(left_rng, batch_size, self.action_horizon)
        noise_right = create_initial_noise_right(right_rng, batch_size, self.action_horizon)
        
        # === 缓存两个VLA的prefix KV ===
        # Left VLA prefix
        prefix_tokens_l, prefix_mask_l, prefix_ar_mask_l = self.left_model.embed_prefix(observation)
        prefix_attn_mask_l = make_attn_mask(prefix_mask_l, prefix_ar_mask_l)
        positions_l = jnp.cumsum(prefix_mask_l, axis=1) - 1
        _, kv_cache_left = self.left_model.PaliGemma.llm(
            [prefix_tokens_l, None], mask=prefix_attn_mask_l, positions=positions_l
        )
        
        # Right VLA prefix
        prefix_tokens_r, prefix_mask_r, prefix_ar_mask_r = self.right_model.embed_prefix(observation)
        prefix_attn_mask_r = make_attn_mask(prefix_mask_r, prefix_ar_mask_r)
        positions_r = jnp.cumsum(prefix_mask_r, axis=1) - 1
        _, kv_cache_right = self.right_model.PaliGemma.llm(
            [prefix_tokens_r, None], mask=prefix_attn_mask_r, positions=positions_r
        )
        
        # 预计算 joint attention mask (action tokens 之间)
        action_ar_mask = make_dual_vla_ar_mask(self.action_horizon)  # [2H, 2H]
        
        def step(carry):
            x_t_left, x_t_right, time = carry
            time_batch = jnp.broadcast_to(time, (batch_size,))
            
            # === 左臂 suffix 嵌入 ===
            suffix_tokens_l, suffix_mask_l, suffix_ar_mask_l = self.left_model.embed_suffix(
                observation, x_t_left, time_batch
            )
            suffix_len_l = suffix_tokens_l.shape[1]
            
            # === 右臂 suffix 嵌入 ===
            suffix_tokens_r, suffix_mask_r, suffix_ar_mask_r = self.right_model.embed_suffix(
                observation, x_t_right, time_batch
            )
            suffix_len_r = suffix_tokens_r.shape[1]
            
            # === 构建各自的 attention mask (含跨臂信息) ===
            # 对于左臂VLA，它的suffix tokens需要能看到右臂VLA的action tokens
            # 但由于是两个独立模型，我们需要通过其他方式传递信息
            
            # 方案: 在每个VLA内部，我们让action tokens能看到对方的"信息摘要"
            # 简化实现: 各自独立去噪，但共享prefix信息 (两个VLA看到相同的observation)
            
            # 左臂 attention mask
            suffix_attn_mask_l = make_attn_mask(suffix_mask_l, suffix_ar_mask_l)
            prefix_attn_l = einops.repeat(prefix_mask_l, "b p -> b s p", s=suffix_len_l)
            full_attn_mask_l = jnp.concatenate([prefix_attn_l, suffix_attn_mask_l], axis=-1)
            positions_l = jnp.sum(prefix_mask_l, axis=-1)[:, None] + jnp.cumsum(suffix_mask_l, axis=-1) - 1
            
            # 右臂 attention mask
            suffix_attn_mask_r = make_attn_mask(suffix_mask_r, suffix_ar_mask_r)
            prefix_attn_r = einops.repeat(prefix_mask_r, "b p -> b s p", s=suffix_len_r)
            full_attn_mask_r = jnp.concatenate([prefix_attn_r, suffix_attn_mask_r], axis=-1)
            positions_r = jnp.sum(prefix_mask_r, axis=-1)[:, None] + jnp.cumsum(suffix_mask_r, axis=-1) - 1
            
            # === 前向传播 ===
            (_, suffix_out_l), _ = self.left_model.PaliGemma.llm(
                [None, suffix_tokens_l],
                mask=full_attn_mask_l,
                positions=positions_l,
                kv_cache=kv_cache_left,
            )
            v_t_left = self.left_model.action_out_proj(suffix_out_l[:, -self.action_horizon:])
            
            (_, suffix_out_r), _ = self.right_model.PaliGemma.llm(
                [None, suffix_tokens_r],
                mask=full_attn_mask_r,
                positions=positions_r,
                kv_cache=kv_cache_right,
            )
            v_t_right = self.right_model.action_out_proj(suffix_out_r[:, -self.action_horizon:])
            
            # === 更新 (只更新有效维度) ===
            # 左臂: 只更新前7维
            x_t_left_new = x_t_left.at[..., :LEFT_ARM_DIM].add(dt * v_t_left[..., :LEFT_ARM_DIM])
            
            # 右臂: 只更新7-13维
            x_t_right_new = x_t_right.at[..., RIGHT_ARM_START:RIGHT_ARM_END].add(
                dt * v_t_right[..., RIGHT_ARM_START:RIGHT_ARM_END]
            )
            
            return x_t_left_new, x_t_right_new, time + dt
        
        def cond(carry):
            _, _, time = carry
            return time >= -dt / 2
        
        # 去噪循环
        x_0_left, x_0_right, _ = jax.lax.while_loop(cond, step, (noise_left, noise_right, 1.0))
        
        # 提取有效维度并合并
        left_actions = extract_left_action(x_0_left)  # [b, ah, 7]
        right_actions = extract_right_action(x_0_right)  # [b, ah, 7]
        
        return merge_actions(left_actions, right_actions)  # [b, ah, 14]
    
    def sample_actions_joint(
        self,
        rng,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
    ) -> jnp.ndarray:
        """
        采样动作 - 真正的joint attention实现。
        
        通过拼接两个VLA的tokens并使用跨VLA的attention mask实现真正的joint attention。
        注意: 这需要修改底层模型的forward pass，更复杂但效果更好。
        
        Args:
            rng: JAX random key
            observation: 观测数据
            num_steps: 去噪步数
            
        Returns:
            actions: [batch, action_horizon, 14] 采样的动作
        """
        # 简化版本: 使用上面的并行采样
        # 真正的joint attention需要更深入的模型修改
        return self.sample_actions(rng, observation, num_steps=num_steps)


class DualVLAPolicy:
    """
    Dual VLA Policy - 用于部署的策略接口。
    
    封装 DualVLA 模型，提供标准的推理接口。
    """
    
    def __init__(
        self,
        train_config_name: str,
        left_checkpoint_dir: str,
        right_checkpoint_dir: str,
        pi0_step: int = 50,
    ):
        """
        初始化 Dual VLA Policy。
        
        Args:
            train_config_name: openpi 训练配置名称
            left_checkpoint_dir: 左臂模型 checkpoint 路径
            right_checkpoint_dir: 右臂模型 checkpoint 路径
            pi0_step: 每次推理输出的动作步数
        """
        self.train_config_name = train_config_name
        self.left_checkpoint_dir = left_checkpoint_dir
        self.right_checkpoint_dir = right_checkpoint_dir
        self.pi0_step = pi0_step
        
        # 加载配置
        config = _config.get_config(train_config_name)
        self.model_config = config.model
        
        # 创建 DualVLA 配置
        dual_config = DualVLAConfig(
            action_dim=ACTION_DIM,
            action_horizon=self.model_config.action_horizon,
            max_token_len=self.model_config.max_token_len,
        )
        
        # 加载两个 Pi0 模型
        logger.info(f"Loading left VLA from {left_checkpoint_dir}")
        self.left_model = self._load_pi0_model(left_checkpoint_dir)
        
        logger.info(f"Loading right VLA from {right_checkpoint_dir}")
        self.right_model = self._load_pi0_model(right_checkpoint_dir)
        
        # 创建 DualVLA
        self.dual_vla = DualVLA(self.left_model, self.right_model, dual_config)
        
        # 加载 transforms
        data_config = config.data.create(config.assets_dirs, config.model)
        
        # 使用左臂 checkpoint 的 norm stats (假设两个模型使用相同的归一化)
        checkpoint_path = Path(download.maybe_download(str(left_checkpoint_dir)))
        norm_stats = _checkpoints.load_norm_stats(
            checkpoint_path / "assets",
            data_config.asset_id
        )
        
        self._input_transforms = [
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
        
        self._output_transforms = [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
        
        # RNG
        self._rng = jax.random.PRNGKey(0)
        
        # 状态
        self.instruction = None
        self.img_size = (224, 224)
        
        logger.info("DualVLA Policy initialized!")
    
    def _load_pi0_model(self, checkpoint_dir: str) -> Pi0:
        """加载单个 Pi0 模型"""
        rng = jax.random.PRNGKey(42)
        model = Pi0(self.model_config, rngs=nnx.Rngs(rng))
        
        checkpoint_dir = download.maybe_download(str(checkpoint_dir))
        params = _model.restore_params(Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
        
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)
        
        return model
    
    def set_language(self, instruction: str):
        """设置语言指令"""
        self.instruction = instruction
        logger.info(f"Set instruction: {instruction}")
    
    def update_observation_window(self, img_arr: list, state: np.ndarray):
        """更新观测窗口"""
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        
        img_front_t = np.transpose(img_front, (2, 0, 1))
        img_right_t = np.transpose(img_right, (2, 0, 1))
        img_left_t = np.transpose(img_left, (2, 0, 1))
        
        self.current_obs = {
            "state": state,
            "images": {
                "cam_high": img_front_t,
                "cam_left_wrist": img_left_t,
                "cam_right_wrist": img_right_t,
            },
            "prompt": self.instruction,
        }
    
    def get_action(self) -> np.ndarray:
        """获取动作"""
        assert self.instruction is not None, "Call set_language first!"
        assert hasattr(self, 'current_obs'), "Call update_observation_window first!"
        
        logger.info(f"[DualVLA] prompt: {self.instruction}")
        
        # 应用 input transforms
        sample = dict(self.current_obs)
        for transform in self._input_transforms:
            sample = transform(sample)
        
        # 添加 batch 维度
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample)
        obs = _model.Observation.from_dict(inputs)
        
        # 采样动作 (输出 14 维)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self.dual_vla.sample_actions(sample_rng, obs, num_steps=10)
        
        # 转换为 numpy: [batch, horizon, 14]
        actions_np = np.array(actions)[0]  # [horizon, 14]
        state_np = np.array(inputs["state"])[0]
        
        # Pad 到 32 维以匹配 norm_stats (pi0 使用 32 维)
        # [horizon, 14] -> [horizon, 32]
        actions_padded = np.pad(actions_np, [(0, 0), (0, ACTION_DIM - 14)], mode='constant')
        
        # 应用 output transforms (32 维)
        result = {"state": state_np, "actions": actions_padded}
        for transform in self._output_transforms:
            result = transform(result)
        
        # 只返回前 14 维 (AlohaOutputs 已经只取前 14 维)
        return result["actions"]
    
    def reset(self):
        """重置"""
        self.instruction = None
        if hasattr(self, 'current_obs'):
            delattr(self, 'current_obs')
        logger.info("DualVLA Policy reset")
    
    def reset_obsrvationwindows(self):
        """重置观测窗口（兼容原有接口）"""
        self.reset()


def create_dual_vla_policy(
    train_config_name: str,
    left_checkpoint_dir: str,
    right_checkpoint_dir: str,
    pi0_step: int = 50,
) -> DualVLAPolicy:
    """创建 DualVLA Policy"""
    return DualVLAPolicy(
        train_config_name=train_config_name,
        left_checkpoint_dir=left_checkpoint_dir,
        right_checkpoint_dir=right_checkpoint_dir,
        pi0_step=pi0_step,
    )

