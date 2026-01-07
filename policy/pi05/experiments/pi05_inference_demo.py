#!/usr/bin/env python3
"""
pi0.5 双臂任务推理示例

直接使用 pi0.5 预训练模型进行推理，输入双臂任务 prompt，输出两只手臂的协调动作。

Usage:
    cd /data0/users/haoce/RoboTwin/policy/pi05
    
    # 使用预训练模型（自动下载）
    python experiments/pi05_inference_demo.py --use-pretrained
    
    # 使用自定义检查点
    python experiments/pi05_inference_demo.py --checkpoint <checkpoint_path>
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import h5py
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import download


# 预训练模型路径
PRETRAINED_MODELS = {
    "pi05_base": "s3://openpi-assets/checkpoints/pi05_base/params",
    "pi0_base": "s3://openpi-assets/checkpoints/pi0_base/params",
    "pi0_fast_base": "s3://openpi-assets/checkpoints/pi0_fast_base/params",
}


class Pi05Inference:
    """pi0.5 推理封装类"""
    
    def __init__(self, config_name: str, checkpoint_path: str = None, use_pretrained: bool = False):
        """
        Args:
            config_name: 配置名称，如 "pi05_aloha_full_base"
            checkpoint_path: 检查点路径（可选）
            use_pretrained: 是否使用预训练模型
        """
        print(f"Loading pi0.5 model...")
        print(f"  Config: {config_name}")
        
        # 获取检查点路径
        if use_pretrained:
            # 使用预训练模型
            pretrained_url = PRETRAINED_MODELS.get("pi05_base")
            print(f"  Using pretrained model: {pretrained_url}")
            print(f"  Downloading... (this may take a while on first run)")
            checkpoint_path = download.maybe_download(pretrained_url)
        
        print(f"  Checkpoint: {checkpoint_path}")
        
        self.config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            self.config,
            checkpoint_path,
        )
        print("Model loaded successfully!")
        
        # 动作维度信息
        self.action_dim = 32  # 左臂16维 + 右臂16维
        self.action_horizon = 50  # 预测50步动作
        
    def infer(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str,
    ) -> dict:
        """
        执行推理
        
        Args:
            images: 图像字典，键为相机名称，值为图像数组 (H, W, 3)
                   期望的键: "cam_high", "cam_left_wrist", "cam_right_wrist"
            state: 机器人状态 (32,) - 左臂关节+右臂关节
            prompt: 任务描述，如 "Grab the red block with left arm and pass to right arm"
            
        Returns:
            dict: {
                "actions": (50, 32) 动作序列，前16维是左臂，后16维是右臂
                "left_arm_actions": (50, 16) 左臂动作
                "right_arm_actions": (50, 16) 右臂动作
            }
        """
        # 准备图像 - 转换为 (C, H, W) 格式
        processed_images = {}
        for name, img in images.items():
            if img.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
                img = np.transpose(img, (2, 0, 1))
            processed_images[name] = img
        
        # 构建观测
        observation = {
            "state": state,
            "images": processed_images,
            "prompt": prompt,
        }
        
        # 执行推理
        print(f"[pi0.5] Prompt: {prompt}")
        result = self.policy.infer(observation)
        actions = result["actions"]  # (50, 32)
        
        # 分离左右臂动作
        left_arm_actions = actions[:, :16]   # 前16维
        right_arm_actions = actions[:, 16:]  # 后16维
        
        return {
            "actions": actions,
            "left_arm_actions": left_arm_actions,
            "right_arm_actions": right_arm_actions,
        }


def load_demo_data(data_dir: str, episode_idx: int = 0) -> dict:
    """加载演示数据用于测试"""
    data_path = Path(data_dir) / "data" / f"episode{episode_idx}.hdf5"
    instruction_path = Path(data_dir) / "instructions" / f"episode{episode_idx}.json"
    
    print(f"Loading demo data from {data_path}")
    
    images = {}
    state = None
    
    with h5py.File(data_path, 'r') as f:
        # 尝试加载图像
        for key in ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 
                    'observation.images.cam_high', 'images/cam_high']:
            for k in f.keys():
                if 'image' in k.lower() or 'cam' in k.lower():
                    data = np.array(f[k])
                    if len(data.shape) == 4:  # (T, H, W, C)
                        images[k] = data[0]
                    elif len(data.shape) == 3 and data.shape[-1] == 3:
                        images[k] = data
        
        # 尝试加载状态
        for key in ['qpos', 'state', 'observation.state', 'puppet_arm']:
            if key in f:
                data = np.array(f[key])
                if len(data.shape) == 2:
                    state = data[0]
                else:
                    state = data
                break
    
    # 加载指令
    instructions = []
    if instruction_path.exists():
        with open(instruction_path, 'r') as f:
            instr_data = json.load(f)
            if isinstance(instr_data, dict):
                if 'seen' in instr_data:
                    instructions = instr_data['seen']
                elif 'unseen' in instr_data:
                    instructions = instr_data['unseen']
            elif isinstance(instr_data, list):
                instructions = instr_data
    
    return {
        'images': images,
        'state': state,
        'instructions': instructions,
    }


def create_dummy_data():
    """创建假数据用于测试"""
    print("Creating dummy data for testing...")
    
    # 创建假图像 (224, 224, 3)
    images = {
        "cam_high": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    }
    
    # 创建假状态 (32,) - 左臂16维 + 右臂16维
    state = np.zeros(32, dtype=np.float32)
    
    return images, state


def main():
    parser = argparse.ArgumentParser(description="pi0.5 Inference Demo")
    parser.add_argument("--config", type=str, default="pi05_aloha_full_base",
                        help="Config name (default: pi05_aloha_full_base)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (optional if using --use-pretrained)")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="Use pretrained pi0.5 model (auto download from S3)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to demo data directory (optional)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to load")
    parser.add_argument("--prompt", type=str, 
                        default="Grab the red block with the left arm and pass it to the right arm",
                        help="Task prompt")
    args = parser.parse_args()
    
    # 检查参数
    if not args.use_pretrained and not args.checkpoint:
        print("Error: Please specify --use-pretrained or --checkpoint")
        print("\nExamples:")
        print("  python experiments/pi05_inference_demo.py --use-pretrained")
        print("  python experiments/pi05_inference_demo.py --checkpoint /path/to/checkpoint")
        return None
    
    # 创建推理器
    inferencer = Pi05Inference(args.config, args.checkpoint, args.use_pretrained)
    
    # 准备数据
    if args.data_dir and Path(args.data_dir).exists():
        data = load_demo_data(args.data_dir, args.episode)
        images = data['images']
        state = data['state']
        if data['instructions']:
            args.prompt = data['instructions'][0]
    else:
        images, state = create_dummy_data()
    
    # 确保有足够的图像
    required_cams = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for cam in required_cams:
        if cam not in images:
            # 用已有的图像填充
            if images:
                images[cam] = list(images.values())[0]
            else:
                images[cam] = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 确保状态维度正确
    if state is None:
        state = np.zeros(32, dtype=np.float32)
    elif len(state) < 32:
        state = np.pad(state, (0, 32 - len(state)))
    elif len(state) > 32:
        state = state[:32]
    
    print("\n" + "=" * 60)
    print("pi0.5 Inference Demo")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Images: {list(images.keys())}")
    print(f"State shape: {state.shape}")
    print("=" * 60)
    
    # 执行推理
    result = inferencer.infer(images, state, args.prompt)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Full actions shape: {result['actions'].shape}")
    print(f"Left arm actions shape: {result['left_arm_actions'].shape}")
    print(f"Right arm actions shape: {result['right_arm_actions'].shape}")
    
    print("\n--- First 5 steps of actions ---")
    print("\nLeft Arm (first 5 steps, first 6 dims):")
    print(result['left_arm_actions'][:5, :6])
    
    print("\nRight Arm (first 5 steps, first 6 dims):")
    print(result['right_arm_actions'][:5, :6])
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()

