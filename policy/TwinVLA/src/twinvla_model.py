"""
TwinVLA Model - 双臂 Vision-Language-Action 模型，带 Joint Attention

使用 TwinPi0 实现真正的跨臂 Joint Attention：
- 每个时间步的动作被拆成两个 token (L_t, R_t)
- L_t 能看到 L_{<=t} + R_{<t}
- R_t 能看到 R_{<=t} + L_{<=t}
- 通过零填充保持预训练权重兼容
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# 添加 pi05 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pi05" / "src"))

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms
from openpi.shared import download

from .twin_pi0 import TwinPi0, create_twin_pi0_from_checkpoint

logger = logging.getLogger(__name__)


class TwinVLAPolicy:
    """
    TwinVLA Policy - 使用 Joint Attention 的双臂协调推理。
    
    核心改进:
    1. 真正的 Joint Attention: 在 action tokens 层让左右臂互相交流
    2. 不改变模型权重结构: 可以直接加载 Pi0 checkpoint
    3. 通过 interleave + 零填充实现 token 重组
    """
    
    def __init__(
        self,
        train_config_name: str,
        checkpoint_dir: str,
        pi0_step: int = 50,
    ):
        """
        初始化 TwinVLA Policy。
        """
        self.train_config_name = train_config_name
        self.checkpoint_dir = checkpoint_dir
        self.pi0_step = pi0_step
        
        # 加载训练配置
        config = _config.get_config(train_config_name)
        self.model_config = config.model
        
        # 创建 TwinPi0 模型并加载权重
        logger.info(f"Creating TwinPi0 model from {checkpoint_dir}")
        self.model = create_twin_pi0_from_checkpoint(self.model_config, checkpoint_dir)
        
        # 加载 data transforms
        data_config = config.data.create(config.assets_dirs, config.model)
        
        # 加载 norm stats
        checkpoint_path = Path(download.maybe_download(str(checkpoint_dir)))
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
        
        # 图像尺寸
        self.img_size = (224, 224)
        
        # 语言指令
        self.instruction = None
        
        # RNG for sampling
        self._rng = jax.random.PRNGKey(0)
        
        logger.info("TwinVLA Policy initialized with Joint Attention!")
    
    def set_language(self, instruction: str):
        """设置语言指令"""
        self.instruction = instruction
        logger.info(f"Set instruction: {instruction}")
    
    def update_observation_window(self, img_arr: list, state: np.ndarray):
        """更新观测窗口"""
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        
        # 转换图像格式 [H, W, C] -> [C, H, W]
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
        """
        使用 Joint Attention 获取双臂动作。
        
        Returns:
            actions: [horizon, action_dim] 双臂动作序列
        """
        assert self.instruction is not None, "Call set_language first!"
        assert hasattr(self, 'current_obs'), "Call update_observation_window first!"
        
        logger.info(f"[TwinVLA Joint Attention] prompt: {self.instruction}")
        
        # 应用 input transforms
        sample = dict(self.current_obs)
        for transform in self._input_transforms:
            sample = transform(sample)
        
        # 添加 batch 维度并转换为 jax array
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], sample)
        
        # 使用 Observation.from_dict 创建 Observation 对象
        obs = _model.Observation.from_dict(inputs)
        
        # 使用 TwinPi0 的 sample_actions（带 Joint Attention）
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self.model.sample_actions(
            sample_rng,
            obs,
            num_steps=10,
            )
        
        # 转换为 numpy，移除 batch 维度
        actions_np = np.array(actions)[0]
        state_np = np.array(inputs["state"])[0]
        
        # 应用反归一化 (output transforms 需要 state 和 actions)
        result = {
            "state": state_np,
            "actions": actions_np,
        }
        for transform in self._output_transforms:
            result = transform(result)
        
        return result["actions"]
    
    def reset(self):
        """重置 policy 状态"""
        self.instruction = None
        if hasattr(self, 'current_obs'):
            delattr(self, 'current_obs')
        logger.info("TwinVLA Policy reset")
    
    def reset_obsrvationwindows(self):
        """重置观测窗口（兼容原有接口）"""
        self.reset()


def create_twinvla_policy(
    train_config_name: str,
    checkpoint_dir: str,
    pi0_step: int = 50,
) -> TwinVLAPolicy:
    """创建 TwinVLA Policy"""
    return TwinVLAPolicy(
        train_config_name=train_config_name,
        checkpoint_dir=checkpoint_dir,
        pi0_step=pi0_step,
    )
