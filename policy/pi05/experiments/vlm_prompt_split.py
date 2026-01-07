#!/usr/bin/env python3
"""
VLM Prompt Split Experiment

利用 VLM 模型测试其能否将一个双臂任务的 instruction 分解为 left arm prompt 和 right arm prompt。

支持多种 VLM 后端:
- qwen-vl: 使用 Qwen-VL 模型 (推荐，无需特殊认证)
- openai: 使用 GPT-4V API
- simple: 基于规则的简单分解 (用于测试)

Usage:
    cd /data0/users/haoce/RoboTwin/policy/pi05
    python experiments/vlm_prompt_split.py --backend simple  # 快速测试
    python experiments/vlm_prompt_split.py --backend qwen-vl  # 使用 Qwen-VL
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLMBackend(ABC):
    """VLM 后端抽象基类。"""
    
    @abstractmethod
    def generate_arm_prompts(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict[str, str]:
        """生成 left/right arm prompts。"""
        pass


class SimpleRuleBackend(VLMBackend):
    """基于规则的简单分解（用于测试流程）。"""
    
    def generate_arm_prompts(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict[str, str]:
        """使用简单规则分解任务。"""
        instruction_lower = instruction.lower()
        
        left_prompt = ""
        right_prompt = ""
        
        # 基于关键词的简单分解
        if "left arm" in instruction_lower and "right arm" in instruction_lower:
            # 尝试按 arm 分割
            parts = re.split(r'(?:,\s*)?(?:and\s+)?(?:then\s+)?', instruction, flags=re.IGNORECASE)
            for part in parts:
                part_lower = part.lower()
                if "left" in part_lower:
                    left_prompt = part.strip()
                elif "right" in part_lower:
                    right_prompt = part.strip()
        elif "handover" in instruction_lower or "pass" in instruction_lower:
            # 传递任务
            left_prompt = "Grab the object and move to handover position"
            right_prompt = "Receive the object from left arm and place it at target"
        elif "grab" in instruction_lower or "pick" in instruction_lower:
            if "left" in instruction_lower:
                left_prompt = instruction
                right_prompt = "Wait and prepare to receive"
            else:
                left_prompt = "Assist with grasping"
                right_prompt = instruction
        else:
            # 默认平均分配
            left_prompt = f"Left arm: {instruction}"
            right_prompt = f"Right arm: Coordinate with left arm"
        
        return {
            "left_arm_prompt": left_prompt,
            "right_arm_prompt": right_prompt,
            "raw_output": f"[Rule-based decomposition]\nLeft: {left_prompt}\nRight: {right_prompt}",
        }


class QwenVLBackend(VLMBackend):
    """使用 Qwen-VL 模型进行任务分解。"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载 Qwen-VL 模型。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading Qwen-VL model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            fp16=True,
        ).eval()
        
        logger.info("Qwen-VL model loaded!")
        
    def generate_arm_prompts(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict[str, str]:
        """使用 Qwen-VL 生成 arm prompts。"""
        if self.model is None:
            self.load_model()
        
        # 准备图像
        main_image = images.get('cam_high', list(images.values())[0])
        if main_image.dtype == np.uint8:
            pil_image = Image.fromarray(main_image)
        else:
            pil_image = Image.fromarray((main_image * 255).astype(np.uint8))
        
        # 保存临时图像
        temp_path = "/tmp/vlm_temp_image.jpg"
        pil_image.save(temp_path)
        
        # 构建 prompt
        prompt = f"""Look at this image of a bimanual robot workspace.

Task: {instruction}

Please decompose this task into two separate subtasks - one for the left arm and one for the right arm.

Format your response as:
Left Arm: [subtask for left arm]
Right Arm: [subtask for right arm]"""
        
        # 生成
        query = self.tokenizer.from_list_format([
            {'image': temp_path},
            {'text': prompt},
        ])
        
        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        
        # 解析响应
        result = self._parse_response(response)
        result['raw_output'] = response
        
        return result
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """解析模型响应。"""
        left_prompt = ""
        right_prompt = ""
        
        lines = response.strip().split('\n')
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith('left arm:'):
                left_prompt = line.split(':', 1)[-1].strip()
            elif line_lower.startswith('right arm:'):
                right_prompt = line.split(':', 1)[-1].strip()
        
        # 如果没有找到格式化的输出，尝试其他方式解析
        if not left_prompt and not right_prompt:
            if "Left Arm" in response:
                parts = response.split("Right Arm")
                left_prompt = parts[0].replace("Left Arm", "").strip(": \n")
                if len(parts) > 1:
                    right_prompt = parts[1].strip(": \n")
        
        return {
            "left_arm_prompt": left_prompt,
            "right_arm_prompt": right_prompt,
        }


class InternVLBackend(VLMBackend):
    """使用 InternVL 模型进行任务分解（另一个开源选择）。"""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载 InternVL 模型。"""
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"Loading InternVL model: {self.model_name}")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        logger.info("InternVL model loaded!")
        
    def generate_arm_prompts(
        self,
        images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict[str, str]:
        """使用 InternVL 生成 arm prompts。"""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        if self.model is None:
            self.load_model()
        
        # 准备图像
        main_image = images.get('cam_high', list(images.values())[0])
        if main_image.dtype != np.uint8:
            main_image = (main_image * 255).astype(np.uint8)
        pil_image = Image.fromarray(main_image).convert('RGB')
        
        # 图像预处理
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        pixel_values = transform(pil_image).unsqueeze(0).to(torch.bfloat16).to(self.device)
        
        # 构建 prompt
        prompt = f"""<image>
This is an image of a bimanual robot workspace.

Task: {instruction}

Please decompose this task into two separate subtasks:
1. Left Arm: What should the left arm do?
2. Right Arm: What should the right arm do?

Provide clear and concise instructions for each arm."""
        
        # 生成
        generation_config = dict(max_new_tokens=256, do_sample=False)
        response = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            prompt, 
            generation_config
        )
        
        # 解析响应
        result = self._parse_response(response)
        result['raw_output'] = response
        
        return result
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """解析模型响应。"""
        left_prompt = ""
        right_prompt = ""
        
        # 尝试多种解析模式
        patterns = [
            (r'[Ll]eft\s*[Aa]rm[:\s]+([^\n]+)', r'[Rr]ight\s*[Aa]rm[:\s]+([^\n]+)'),
            (r'1\.\s*[Ll]eft[:\s]+([^\n]+)', r'2\.\s*[Rr]ight[:\s]+([^\n]+)'),
        ]
        
        for left_pattern, right_pattern in patterns:
            left_match = re.search(left_pattern, response)
            right_match = re.search(right_pattern, response)
            if left_match:
                left_prompt = left_match.group(1).strip()
            if right_match:
                right_prompt = right_match.group(1).strip()
            if left_prompt or right_prompt:
                break
        
        return {
            "left_arm_prompt": left_prompt,
            "right_arm_prompt": right_prompt,
        }


def load_episode_data(data_dir: str, episode_idx: int) -> dict:
    """加载一个 episode 的数据。"""
    import h5py
    
    data_path = Path(data_dir) / "data" / f"episode{episode_idx}.hdf5"
    instruction_path = Path(data_dir) / "instructions" / f"episode{episode_idx}.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Episode data not found: {data_path}")
    
    # 加载图像数据
    images = {}
    with h5py.File(data_path, 'r') as f:
        logger.debug(f"HDF5 keys: {list(f.keys())}")
        
        def find_images(group, prefix=""):
            """递归查找图像数据。"""
            for key in group.keys():
                full_key = f"{prefix}/{key}" if prefix else key
                item = group[key]
                if hasattr(item, 'keys'):
                    find_images(item, full_key)
                else:
                    try:
                        data = np.array(item)
                        if len(data.shape) >= 3 and data.shape[-1] == 3:
                            # 可能是图像数据
                            if len(data.shape) == 4:
                                images[key] = data[0]  # 取第一帧
                            elif len(data.shape) == 3:
                                images[key] = data
                    except Exception:
                        pass
        
        find_images(f)
    
    # 加载 instruction
    instructions = []
    if instruction_path.exists():
        with open(instruction_path, 'r') as f:
            instr_data = json.load(f)
            if isinstance(instr_data, dict):
                if 'seen' in instr_data and instr_data['seen']:
                    instructions = instr_data['seen'][:5]
                elif 'unseen' in instr_data and instr_data['unseen']:
                    instructions = instr_data['unseen'][:5]
            elif isinstance(instr_data, list):
                instructions = instr_data[:5]
    
    return {
        'images': images,
        'instructions': instructions,
        'episode_idx': episode_idx,
    }


def get_backend(backend_name: str, device: str = "cuda") -> VLMBackend:
    """获取指定的 VLM 后端。"""
    backends = {
        'simple': SimpleRuleBackend,
        'qwen-vl': lambda: QwenVLBackend(device=device),
        'internvl': lambda: InternVLBackend(device=device),
    }
    
    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")
    
    backend = backends[backend_name]
    return backend() if callable(backend) else backend


def main():
    """主函数：运行 VLM prompt 分解实验。"""
    parser = argparse.ArgumentParser(description="VLM Prompt Split Experiment")
    parser.add_argument("--backend", type=str, default="simple",
                        choices=["simple", "qwen-vl", "internvl"],
                        help="VLM backend to use")
    parser.add_argument("--data-dir", type=str, 
                        default="/data0/users/haoce/RoboTwin/data/handover_block/demo_clean",
                        help="Path to data directory")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2],
                        help="Episode indices to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("VLM Prompt Split Experiment")
    logger.info(f"Backend: {args.backend}")
    logger.info("=" * 60)
    
    # 初始化后端
    try:
        backend = get_backend(args.backend, args.device)
        logger.info(f"Backend initialized: {type(backend).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        return
    
    # 测试多个 episode
    results = []
    
    for ep_idx in args.episodes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Episode {ep_idx}")
        logger.info("=" * 60)
        
        try:
            episode_data = load_episode_data(args.data_dir, ep_idx)
            
            if not episode_data['images']:
                logger.warning(f"No images found for episode {ep_idx}")
                continue
                
            if not episode_data['instructions']:
                logger.warning(f"No instructions found for episode {ep_idx}")
                continue
            
            logger.info(f"Found images: {list(episode_data['images'].keys())}")
            logger.info(f"Found {len(episode_data['instructions'])} instructions")
            
            # 对每个指令进行测试
            for i, instruction in enumerate(episode_data['instructions'][:2]):
                logger.info(f"\n--- Instruction {i+1} ---")
                logger.info(f"Original: {instruction}")
                
                result = backend.generate_arm_prompts(
                    images=episode_data['images'],
                    instruction=instruction,
                )
                
                logger.info(f"Left Arm Prompt: {result['left_arm_prompt']}")
                logger.info(f"Right Arm Prompt: {result['right_arm_prompt']}")
                
                results.append({
                    'episode': ep_idx,
                    'instruction': instruction,
                    **result,
                })
                
        except Exception as e:
            logger.error(f"Error processing episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for r in results:
        logger.info(f"\nEpisode {r['episode']}:")
        logger.info(f"  Instruction: {r['instruction'][:60]}...")
        logger.info(f"  Left:  {r['left_arm_prompt'][:60]}..." if r['left_arm_prompt'] else "  Left:  [empty]")
        logger.info(f"  Right: {r['right_arm_prompt'][:60]}..." if r['right_arm_prompt'] else "  Right: [empty]")
    
    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
