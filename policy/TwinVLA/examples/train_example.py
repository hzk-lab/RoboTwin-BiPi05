#!/usr/bin/env python3
"""
TwinVLA Training Script

使用 shake_bottle 数据训练 TwinVLA 模型，实现从单臂摇瓶子到双臂同时摇瓶子的泛化。

关键特点：
1. 使用两个预训练的 Pi0.5 模型（左臂和右臂）
2. 通过 Joint Attention 实现双臂协调
3. 从单臂数据创建双臂训练样本（镜像）

Usage:
    # 从预训练的 Pi0.5 模型开始训练
    python train.py --pretrained_checkpoint /path/to/pi05/checkpoint
    
    # 从已训练的 PyTorch checkpoint 加载
    python train.py --pytorch_checkpoint /path/to/model.safetensors
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pi05" / "src"))

from src.twinvla_model import (
    TwinVLAModel, 
    TwinVLAConfig,
    create_twinvla_from_pretrained,
    create_twinvla_from_pi05_model,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ShakeBottleBimanualDataset(Dataset):
    """
    将单臂 shake_bottle 数据转换为双臂数据集。
    
    策略：左右臂执行相同的动作（镜像），模拟双臂同时摇两个瓶子。
    """
    
    def __init__(
        self,
        data_dir: str,
        max_episodes: int = 50,
        action_horizon: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon
        
        # 加载所有 episode
        self.episodes = []
        data_path = self.data_dir / "data"
        
        episode_files = sorted(data_path.glob("episode*.hdf5"))[:max_episodes]
        logger.info(f"Loading {len(episode_files)} episodes from {data_path}")
        
        for ep_file in episode_files:
            try:
                with h5py.File(ep_file, 'r') as f:
                    episode_data = self._load_episode(f)
                    if episode_data is not None:
                        self.episodes.append(episode_data)
            except Exception as e:
                logger.warning(f"Failed to load {ep_file}: {e}")
        
        # 加载 instructions
        self.instructions = self._load_instructions()
        
        logger.info(f"Loaded {len(self.episodes)} episodes")
        
        # 计算总样本数
        self.samples = []
        for ep_idx, ep in enumerate(self.episodes):
            num_steps = ep['actions'].shape[0]
            for t in range(0, num_steps - action_horizon, action_horizon // 2):
                self.samples.append((ep_idx, t))
        
        logger.info(f"Total samples: {len(self.samples)}")
    
    def _load_episode(self, f: h5py.File) -> Optional[Dict[str, np.ndarray]]:
        """加载单个 episode 的数据."""
        try:
            data = {}
            
            # 查找图像数据
            def find_arrays(group, prefix=""):
                for key in group.keys():
                    full_key = f"{prefix}/{key}" if prefix else key
                    item = group[key]
                    if hasattr(item, 'keys'):
                        find_arrays(item, full_key)
                    else:
                        arr = np.array(item)
                        if 'cam' in key.lower() or 'image' in key.lower():
                            data[f"images/{key}"] = arr
                        elif 'action' in key.lower():
                            data['actions'] = arr
                        elif 'state' in key.lower():
                            data['state'] = arr
            
            find_arrays(f)
            
            if 'actions' not in data:
                return None
                
            return data
            
        except Exception as e:
            logger.warning(f"Error loading episode: {e}")
            return None
    
    def _load_instructions(self) -> list:
        """加载 instructions."""
        instructions = []
        instr_dir = self.data_dir / "instructions"
        
        if instr_dir.exists():
            for instr_file in sorted(instr_dir.glob("episode*.json")):
                try:
                    with open(instr_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'seen' in data:
                            instructions.append(data['seen'][0] if data['seen'] else "shake the bottle")
                        else:
                            instructions.append("shake the bottle")
                except:
                    instructions.append("shake the bottle")
        
        return instructions
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        
        # 获取图像
        images = {}
        for key, value in ep.items():
            if key.startswith("images/"):
                img_key = key.split("/")[-1]
                img = value[t]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                images[img_key] = torch.from_numpy(img).permute(2, 0, 1)
        
        # 如果只有一个相机，复制为左右 wrist
        if len(images) == 1:
            main_img = list(images.values())[0]
            images = {
                'cam_high': main_img,
                'cam_left_wrist': main_img.clone(),
                'cam_right_wrist': main_img.clone(),
            }
        
        # 获取 actions [horizon, action_dim]
        actions = ep['actions'][t:t+self.action_horizon]
        if len(actions) < self.action_horizon:
            # Padding
            pad_len = self.action_horizon - len(actions)
            actions = np.concatenate([actions, np.repeat(actions[-1:], pad_len, axis=0)])
        
        actions = torch.from_numpy(actions).float()
        
        # 单臂 action -> 双臂 (镜像)
        action_dim = actions.shape[-1]
        if action_dim == 32:
            # 已经是双臂数据
            left_actions = actions[:, :16]
            right_actions = actions[:, 16:]
        else:
            # 单臂数据，复制给双臂
            left_actions = actions
            right_actions = actions.clone()  # 镜像
        
        # 获取 state
        if 'state' in ep:
            state = ep['state'][t]
            state = torch.from_numpy(state).float()
            if len(state) >= 32:
                left_state = state[:16]
                right_state = state[16:32]
            else:
                left_state = state
                right_state = state.clone()
        else:
            left_state = torch.zeros(16)
            right_state = torch.zeros(16)
        
        # 合并 state (双臂共 32 维)
        full_state = torch.cat([left_state, right_state], dim=-1)
        
        # Instruction
        instruction = self.instructions[ep_idx] if ep_idx < len(self.instructions) else "shake the bottle"
        
        return {
            'images': images,
            'state': full_state,
            'left_actions': left_actions,
            'right_actions': right_actions,
            'instruction': instruction,
        }


def collate_fn(batch: list) -> Dict[str, Any]:
    """自定义 collate function."""
    result = {
        'images': {},
        'state': torch.stack([b['state'] for b in batch]),
        'left_actions': torch.stack([b['left_actions'] for b in batch]),
        'right_actions': torch.stack([b['right_actions'] for b in batch]),
        'instructions': [b['instruction'] for b in batch],
    }
    
    # Stack images
    for key in batch[0]['images']:
        result['images'][key] = torch.stack([b['images'][key] for b in batch])
    
    return result


def load_twinvla_model(args, device) -> TwinVLAModel:
    """
    加载 TwinVLA 模型，支持多种方式：
    1. 从预训练的 Pi0.5 checkpoint 初始化
    2. 从已训练的 TwinVLA PyTorch checkpoint 加载
    3. 使用 openpi 的 create_trained_policy 加载 Pi0.5，然后转换
    """
    config = TwinVLAConfig(
        action_dim=16,
        action_horizon=args.action_horizon,
        num_joint_layers=args.num_joint_layers,
    )
    
    if args.pytorch_checkpoint and os.path.exists(args.pytorch_checkpoint):
        # 方式 1: 从已训练的 TwinVLA checkpoint 加载
        logger.info(f"Loading TwinVLA from PyTorch checkpoint: {args.pytorch_checkpoint}")
        model = TwinVLAModel(config)
        checkpoint = torch.load(args.pytorch_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        
    elif args.pretrained_checkpoint:
        # 方式 2: 从预训练的 Pi0.5 checkpoint 初始化
        logger.info(f"Loading TwinVLA from pretrained Pi0.5: {args.pretrained_checkpoint}")
        
        # 检查是否是 PyTorch 格式（model.safetensors）
        safetensors_path = os.path.join(args.pretrained_checkpoint, "model.safetensors")
        
        if os.path.exists(safetensors_path):
            # PyTorch safetensors 格式
            model = create_twinvla_from_pretrained(
                args.pretrained_checkpoint, 
                config=config,
                device=str(device)
            )
        else:
            # 尝试使用 openpi 的方式加载
            try:
                from openpi.policies.policy_config import create_trained_policy
                from openpi.training.config import get_config
                
                # 获取对应的 train_config
                train_config = get_config("pi05_aloha_full_base")
                
                # 加载 policy
                policy = create_trained_policy(
                    train_config, 
                    args.pretrained_checkpoint,
                    pytorch_device=str(device)
                )
                
                # 从 policy 中获取 Pi0.5 模型
                pi05_model = policy._model
                
                # 创建 TwinVLA
                model = create_twinvla_from_pi05_model(
                    pi05_model,
                    config=config,
                    device=str(device)
                )
                logger.info("Successfully loaded Pi0.5 and created TwinVLA")
                
            except Exception as e:
                logger.warning(f"Failed to load via openpi: {e}")
                logger.info("Creating TwinVLA with random initialization")
                model = TwinVLAModel(config)
                model = model.to(device)
    else:
        # 方式 3: 随机初始化
        logger.info("Creating TwinVLA with random initialization")
        model = TwinVLAModel(config)
        model = model.to(device)
    
    return model


def create_observation(batch, tokenizer, device, max_token_len=200):
    """
    从 batch 创建 Observation 对象。
    """
    from openpi.models.model import Observation
    
    # 图像处理
    images = {k: v.to(device) for k, v in batch['images'].items()}
    image_masks = {k: torch.ones(v.shape[0], dtype=torch.bool, device=device) for k, v in images.items()}
    
    # State
    state = batch['state'].to(device)
    
    # Tokenize instructions
    if tokenizer is not None:
        instructions = batch['instructions']
        tokens = tokenizer(
            instructions,
            padding='max_length',
            max_length=max_token_len,
            truncation=True,
            return_tensors='pt'
        )
        tokenized_prompt = tokens['input_ids'].to(device)
        tokenized_prompt_mask = tokens['attention_mask'].bool().to(device)
    else:
        # Dummy tokens
        batch_size = state.shape[0]
        tokenized_prompt = torch.zeros(batch_size, max_token_len, dtype=torch.long, device=device)
        tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool, device=device)
    
    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def train(args):
    """训练 TwinVLA 模型."""
    logger.info("=" * 60)
    logger.info("TwinVLA Training")
    logger.info("=" * 60)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建数据集
    dataset = ShakeBottleBimanualDataset(
        data_dir=args.data_dir,
        max_episodes=args.max_episodes,
        action_horizon=args.action_horizon,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 加载模型
    model = load_twinvla_model(args, device)
    
    # 冻结 VLM backbone，只训练 action expert 和 joint attention
    if args.freeze_vlm:
        logger.info("Freezing VLM backbone...")
        for arm in [model.left_arm, model.right_arm]:
            # 冻结 PaliGemma (VLM backbone)
            if hasattr(arm, 'paligemma_with_expert'):
                for param in arm.paligemma_with_expert.paligemma.parameters():
                    param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(dataloader),
        eta_min=args.learning_rate * 0.01,
    )
    
    # 尝试加载 tokenizer
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer()
        logger.info("Loaded PaliGemma tokenizer")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}, using dummy tokens")
        tokenizer = None
    
    # 训练循环
    logger.info(f"Starting training for {args.num_epochs} epochs")
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_left_loss = 0.0
        total_right_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # 创建 Observation
            observation = create_observation(batch, tokenizer, device)
            
            # 获取 actions
            left_actions = batch['left_actions'].to(device)
            right_actions = batch['right_actions'].to(device)
            
            # Forward
            optimizer.zero_grad()
            
            if args.use_joint_attention:
                left_loss, right_loss = model.forward_with_joint_attention(
                    observation,
                    left_actions,
                    right_actions,
                )
            else:
                left_loss, right_loss = model(
                    observation,
                    left_actions,
                    right_actions,
                )
            
            # 合并 loss
            left_loss_mean = left_loss.mean()
            right_loss_mean = right_loss.mean()
            loss = left_loss_mean + right_loss_mean
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_left_loss += left_loss_mean.item()
            total_right_loss += right_loss_mean.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'L': f'{left_loss_mean.item():.4f}',
                'R': f'{right_loss_mean.item():.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_left_loss = total_left_loss / len(dataloader)
        avg_right_loss = total_right_loss / len(dataloader)
        
        logger.info(
            f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}, "
            f"left = {avg_left_loss:.4f}, right = {avg_right_loss:.4f}"
        )
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(args.output_dir) / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model.config,
            }, save_path)
            logger.info(f"Saved best model to {save_path}")
        
        # 定期保存 checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': model.config,
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
    
    # 保存最终模型
    final_path = Path(args.output_dir) / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, final_path)
    logger.info(f"Saved final model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TwinVLA model")
    
    # Data
    parser.add_argument("--data_dir", type=str, 
                        default="/data0/users/haoce/RoboTwin/data/shake_bottle/demo_clean",
                        help="Path to training data")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="Maximum number of episodes to use")
    
    # Model
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="Action horizon")
    parser.add_argument("--num_joint_layers", type=int, default=4,
                        help="Number of joint attention layers")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained Pi0.5 checkpoint")
    parser.add_argument("--pytorch_checkpoint", type=str, default=None,
                        help="Path to PyTorch TwinVLA checkpoint to resume from")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--freeze_vlm", action="store_true",
                        help="Freeze VLM backbone (only train action expert and joint attention)")
    parser.add_argument("--use_joint_attention", action="store_true",
                        help="Use joint attention during training")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints/twinvla",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
