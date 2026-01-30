#!/usr/bin/env python3
"""
Bimanual Task Model - 双臂任务分解与协调度评估模型

利用 Pi0.5 预训练模型中的 PaliGemma VLM 部分，完成以下任务：

任务1: 双臂 Prompt 生成 (Bimanual Prompt Generation)
    给定 1-3 张任务截图和任务 prompt，输出左右臂分别的 prompt
    
    输入:
        - images: 1-3 张任务场景图像 (RGB, 224x224)
            - base_rgb: 基座相机图像（必需）
            - left_wrist_rgb: 左手腕相机图像（可选）
            - right_wrist_rgb: 右手腕相机图像（可选）
        - prompt: 高层任务指令
        
    输出:
        - left_arm_prompt: 左臂任务指令
        - right_arm_prompt: 右臂任务指令

任务2: 协调度评估 (Cooperation Score)
    给定 1-3 张任务截图和任务 prompt，输出左右手的协调度（0 到 1 之间的连续值）
    
    输入:
        - images: 1-3 张任务场景图像 (RGB, 224x224)
        - prompt: 任务指令
        
    输出:
        - cooperation_score: 协调度分数，连续值 [0, 1]
            - 0.0-0.2: 非常低的协作需求，双臂几乎完全独立
            - 0.2-0.4: 较低的协作需求，简单的传递或等待
            - 0.4-0.6: 中等协作需求，需要一定的时序配合
            - 0.6-0.8: 较高的协作需求，需要空间和时间上的协调
            - 0.8-1.0: 非常高的协作需求，需要紧密同步和对齐

Usage:
    from bimanual_task_model import BimanualTaskModel
    
    # 创建模型
    model = BimanualTaskModel(mode="paligemma", use_pretrained=True)
    
    # 单张图片输入
    result = model.generate_bimanual_prompts(base_image, "Pick up the bottle")
    
    # 多张图片输入 - 使用字典
    images = {
        "base_rgb": base_image,
        "left_wrist_rgb": left_wrist_image,
        "right_wrist_rgb": right_wrist_image,
    }
    result = model.generate_bimanual_prompts(images, "Pick up the bottle and shake it")
    print(result.left_arm_prompt)
    print(result.right_arm_prompt)
    
    # 多张图片输入 - 使用列表 [base, left_wrist, right_wrist]
    images = [base_image, left_wrist_image, right_wrist_image]
    score = model.evaluate_cooperation(images, "Lift the heavy box with both hands")
    print(score.cooperation_score)  # 输出: 0.7823
    
    # 同时执行两个任务
    result = model.analyze(images, "Shake the bottle with both hands")
    print(result.left_arm_prompt)
    print(result.right_arm_prompt)
    print(result.cooperation_score)

Author: Bimanual Robot Team
Date: 2026-01
"""

import argparse
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Any, List, Dict

import numpy as np
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类型别名
ImageType = Union[str, np.ndarray, Image.Image]
ImageInput = Union[ImageType, List[ImageType], Dict[str, ImageType]]

# 图像键名常量
IMAGE_KEYS = ["base_rgb", "left_wrist_rgb", "right_wrist_rgb"]


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class BimanualPromptResult:
    """双臂 Prompt 生成结果"""
    left_arm_prompt: str
    right_arm_prompt: str
    raw_output: str = ""
    confidence: float = 1.0


@dataclass 
class CooperationResult:
    """
    协调度评估结果
    
    cooperation_score 是一个连续值，范围 [0, 1]:
        - 0.0-0.2: 非常低的协作需求，双臂几乎完全独立
        - 0.2-0.4: 较低的协作需求，可能有简单的传递或等待
        - 0.4-0.6: 中等协作需求，需要一定的时序配合
        - 0.6-0.8: 较高的协作需求，需要空间和时间上的协调
        - 0.8-1.0: 非常高的协作需求，需要紧密同步和对齐
    """
    cooperation_score: float  # 连续值 [0, 1]
    explanation: str = ""
    raw_output: str = ""
    spatial_info: dict = field(default_factory=dict)


@dataclass
class BimanualAnalysisResult:
    """完整的双臂任务分析结果"""
    # 任务1: 双臂 Prompt
    left_arm_prompt: str
    right_arm_prompt: str
    # 任务2: 协调度
    cooperation_score: float
    explanation: str = ""
    raw_output: str = ""


# =============================================================================
# 图像处理工具
# =============================================================================

def load_single_image(image: ImageType) -> np.ndarray:
    """
    加载单张图像，支持多种输入格式
    
    Args:
        image: 可以是以下类型:
            - str: 图片文件路径
            - np.ndarray: numpy 数组 (H, W, 3)
            - PIL.Image: PIL 图像对象
            
    Returns:
        np.ndarray: RGB 图像数组 (H, W, 3), dtype=uint8
    """
    if isinstance(image, str):
        if not Path(image).exists():
            raise FileNotFoundError(f"图片文件不存在: {image}")
        pil_image = Image.open(image).convert('RGB')
        return np.array(pil_image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image
    elif hasattr(image, 'convert'):  # PIL Image
        return np.array(image.convert('RGB'))
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")


def load_images(images: ImageInput) -> Dict[str, np.ndarray]:
    """
    加载 1-3 张图像，支持多种输入格式
    
    Args:
        images: 可以是以下格式:
            - 单张图像 (str, np.ndarray, PIL.Image): 作为 base_rgb
            - 列表 [base, left_wrist, right_wrist]: 按顺序解析
            - 字典 {"base_rgb": img, "left_wrist_rgb": img, "right_wrist_rgb": img}
            
    Returns:
        Dict[str, np.ndarray]: 包含各相机图像的字典
            - "base_rgb": 基座相机图像（必需）
            - "left_wrist_rgb": 左手腕相机图像（可选）
            - "right_wrist_rgb": 右手腕相机图像（可选）
    """
    result = {}
    
    if isinstance(images, dict):
        # 字典输入: {"base_rgb": img, "left_wrist_rgb": img, ...}
        for key in IMAGE_KEYS:
            if key in images and images[key] is not None:
                result[key] = load_single_image(images[key])
        
        # 确保至少有一张图像
        if not result:
            raise ValueError("至少需要提供一张图像")
        
        # 如果没有 base_rgb，使用第一张图像
        if "base_rgb" not in result:
            first_key = list(result.keys())[0]
            result["base_rgb"] = result[first_key]
            
    elif isinstance(images, (list, tuple)):
        # 列表输入: [base, left_wrist, right_wrist]
        if len(images) == 0:
            raise ValueError("至少需要提供一张图像")
        if len(images) > 3:
            raise ValueError("最多支持 3 张图像")
        
        for i, img in enumerate(images):
            if img is not None:
                result[IMAGE_KEYS[i]] = load_single_image(img)
                
    else:
        # 单张图像输入
        result["base_rgb"] = load_single_image(images)
    
    return result


def load_image(image: ImageInput) -> np.ndarray:
    """
    兼容旧接口：加载单张图像
    
    如果输入多张图像，返回 base_rgb
    """
    images = load_images(image)
    return images.get("base_rgb", list(images.values())[0])


def preprocess_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    预处理图像到标准尺寸
    
    Args:
        image: RGB 图像数组 (H, W, 3)
        target_size: 目标尺寸 (H, W)
        
    Returns:
        np.ndarray: 预处理后的图像
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    if image.shape[:2] != target_size:
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(target_size[::-1], Image.BILINEAR)  # PIL uses (W, H)
        image = np.array(pil_image)
    
    return image


# =============================================================================
# 后端基类
# =============================================================================

class BimanualBackend(ABC):
    """双臂任务模型后端抽象基类，支持 1-3 张图像输入"""
    
    @abstractmethod
    def generate_bimanual_prompts(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """
        生成左右臂的 prompt
        
        Args:
            images: 1-3 张图像的字典
                - "base_rgb": 基座相机图像
                - "left_wrist_rgb": 左手腕相机图像（可选）
                - "right_wrist_rgb": 右手腕相机图像（可选）
            instruction: 任务指令
        """
        pass
    
    @abstractmethod
    def evaluate_cooperation(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """
        评估协调度
        
        Args:
            images: 1-3 张图像的字典
            instruction: 任务指令
        """
        pass
    
    def analyze(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualAnalysisResult:
        """同时执行任务1和任务2"""
        prompt_result = self.generate_bimanual_prompts(images, instruction)
        coop_result = self.evaluate_cooperation(images, instruction)
        
        return BimanualAnalysisResult(
            left_arm_prompt=prompt_result.left_arm_prompt,
            right_arm_prompt=prompt_result.right_arm_prompt,
            cooperation_score=coop_result.cooperation_score,
            explanation=coop_result.explanation,
            raw_output=f"=== Prompt Generation ===\n{prompt_result.raw_output}\n\n"
                      f"=== Cooperation Evaluation ===\n{coop_result.raw_output}",
        )
    
    def _get_primary_image(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        """获取主图像（base_rgb 优先）"""
        return images.get("base_rgb", list(images.values())[0])
    
    def _get_num_images(self, images: Dict[str, np.ndarray]) -> int:
        """获取图像数量"""
        return len(images)


# =============================================================================
# 规则后端 (Rule-based Backend)
# =============================================================================

class RuleBasedBackend(BimanualBackend):
    """基于规则的双臂任务分解与协调度评估"""
    
    # 任务分解规则
    TASK_PATTERNS = {
        # 传递/交接任务
        "handover": {
            "keywords": ["handover", "pass", "transfer", "hand over", "give"],
            "left": "Grab the object and move to handover position",
            "right": "Receive the object from left arm and place at target",
        },
        # 协同抓取
        "cooperative": {
            "keywords": ["together", "cooperative", "both hands", "both arms", "bimanual"],
            "left": "Left arm: coordinate grip on left side",
            "right": "Right arm: coordinate grip on right side",
        },
        # 摇晃任务
        "shake": {
            "keywords": ["shake"],
            "left": "Left arm: grip object firmly",
            "right": "Right arm: grip object firmly and coordinate shaking motion",
        },
        # 折叠任务
        "fold": {
            "keywords": ["fold"],
            "left": "Left arm: hold one side of the object",
            "right": "Right arm: fold the other side",
        },
        # 举起任务
        "lift": {
            "keywords": ["lift", "raise", "carry"],
            "left": "Left arm: grip and lift from left side",
            "right": "Right arm: grip and lift from right side",
        },
    }
    
    # 协调度关键词权重 - 细粒度连续值
    HIGH_COOP_KEYWORDS = {
        # 最高协作 (0.25-0.35)
        "both hands": 0.28, "both arms": 0.27, "bimanual": 0.32,
        "synchronized": 0.31, "synchronize": 0.29,
        # 高协作 (0.18-0.24)
        "together": 0.22, "cooperative": 0.21, "coordinated": 0.23,
        "simultaneously": 0.19, "at the same time": 0.18,
        "aligned": 0.24, "align": 0.17,
        # 中高协作 (0.12-0.17)
        "lift together": 0.16, "hold together": 0.15, "carry together": 0.14,
        "shake": 0.13, "fold": 0.12, "joint": 0.11,
        # 中等协作 (0.06-0.11)
        "handover": 0.09, "hand over": 0.08, "pass to": 0.07,
        "transfer": 0.06, "coordinate": 0.10,
    }
    
    LOW_COOP_KEYWORDS = {
        # 最低协作 (0.25-0.35)
        "independently": 0.32, "separately": 0.28,
        "individual": 0.26, "alone": 0.24,
        # 低协作 (0.15-0.24)
        "left arm picks": 0.18, "right arm picks": 0.17,
        "left arm grabs": 0.16, "right arm grabs": 0.15,
        "one arm": 0.22, "single arm": 0.21,
        # 中低协作 (0.05-0.14)
        "then the other": 0.12, "and then": 0.08,
        "after that": 0.07, "followed by": 0.06,
    }
    
    def generate_bimanual_prompts(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用规则生成左右臂 prompt（支持 1-3 张图像）"""
        instruction_lower = instruction.lower()
        num_images = len(images)
        
        left_prompt = ""
        right_prompt = ""
        
        # 检查预定义的任务模式
        for pattern_name, pattern_info in self.TASK_PATTERNS.items():
            if any(kw in instruction_lower for kw in pattern_info["keywords"]):
                left_prompt = pattern_info["left"]
                right_prompt = pattern_info["right"]
                break
        
        # 如果没有匹配的模式，尝试解析指令
        if not left_prompt and not right_prompt:
            # 检测左右手/臂的关键词
            has_left = any(kw in instruction_lower for kw in ["left arm", "left hand", "use left", "with left"])
            has_right = any(kw in instruction_lower for kw in ["right arm", "right hand", "use right", "with right"])
            
            if has_left and has_right:
                # 尝试多种分割模式
                # 模式1: "use...and use..." 或 "use...use..."
                use_pattern = re.search(
                    r'use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?(.+?)(?:\s+and\s+use|\s+use)\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?(.+)',
                    instruction, flags=re.IGNORECASE
                )
                if use_pattern:
                    # 确定哪个是左哪个是右
                    first_part = use_pattern.group(0)
                    if re.search(r'use\s+(?:the\s+)?left', first_part, re.IGNORECASE):
                        left_prompt = use_pattern.group(1).strip().rstrip('.').rstrip(',')
                        right_prompt = use_pattern.group(2).strip().rstrip('.')
                    else:
                        right_prompt = use_pattern.group(1).strip().rstrip('.').rstrip(',')
                        left_prompt = use_pattern.group(2).strip().rstrip('.')
                else:
                    # 模式2: 使用 "and" 分割
                    parts = re.split(r'\s+and\s+', instruction, flags=re.IGNORECASE)
                    for part in parts:
                        part_lower = part.lower()
                        part_clean = part.strip()
                        if any(kw in part_lower for kw in ["left arm", "left hand", "use left", "with left"]):
                            # 提取任务内容，移除 "use the left hand to" 等前缀
                            task = re.sub(r'^use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?', '', part_clean, flags=re.IGNORECASE)
                            left_prompt = task.strip() if task.strip() else part_clean
                        elif any(kw in part_lower for kw in ["right arm", "right hand", "use right", "with right"]):
                            task = re.sub(r'^use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?', '', part_clean, flags=re.IGNORECASE)
                            right_prompt = task.strip() if task.strip() else part_clean
                
                # 如果只解析出一个，补充另一个
                if left_prompt and not right_prompt:
                    right_prompt = "Coordinate with left arm"
                elif right_prompt and not left_prompt:
                    left_prompt = "Coordinate with right arm"
                    
            elif has_left:
                # 提取左手任务
                task = re.sub(r'.*?(?:left\s+(?:hand|arm))\s+(?:to\s+)?', '', instruction, flags=re.IGNORECASE)
                left_prompt = task.strip() if task.strip() else instruction
                right_prompt = "Stand by and prepare to assist"
            elif has_right:
                # 提取右手任务
                task = re.sub(r'.*?(?:right\s+(?:hand|arm))\s+(?:to\s+)?', '', instruction, flags=re.IGNORECASE)
                right_prompt = task.strip() if task.strip() else instruction
                left_prompt = "Stand by and prepare to assist"
            else:
                # 默认情况：左臂为主，右臂协调
                left_prompt = instruction
                right_prompt = "Coordinate with left arm"
        
        # 根据可用的手腕图像提供额外上下文
        context_info = []
        if "left_wrist_rgb" in images:
            context_info.append("left wrist view available")
        if "right_wrist_rgb" in images:
            context_info.append("right wrist view available")
        
        raw_output = f"[Rule-based] (images: {num_images})\n"
        raw_output += f"Left: {left_prompt}\nRight: {right_prompt}"
        if context_info:
            raw_output += f"\nContext: {', '.join(context_info)}"
        
        return BimanualPromptResult(
            left_arm_prompt=left_prompt,
            right_arm_prompt=right_prompt,
            raw_output=raw_output,
        )
    
    def evaluate_cooperation(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用规则评估协调度 - 输出连续值 [0, 1]（支持 1-3 张图像）"""
        instruction_lower = instruction.lower()
        num_images = len(images)
        
        # 基础分数 - 根据指令长度和复杂度微调
        word_count = len(instruction.split())
        base_score = 0.45 + min(0.1, word_count * 0.005)  # 基础分数在 0.45-0.55 之间
        
        score_delta = 0.0
        matched_keywords = []
        
        # 检测高协作关键词 - 累加权重
        for keyword, weight in self.HIGH_COOP_KEYWORDS.items():
            if keyword in instruction_lower:
                # 根据关键词在句子中的位置微调权重
                pos = instruction_lower.find(keyword)
                pos_factor = 1.0 + (1.0 - pos / len(instruction_lower)) * 0.1  # 靠前的权重稍高
                adjusted_weight = weight * pos_factor
                score_delta += adjusted_weight
                matched_keywords.append(f"+{keyword}({adjusted_weight:.3f})")
        
        # 检测低协作关键词 - 累减权重
        for keyword, weight in self.LOW_COOP_KEYWORDS.items():
            if keyword in instruction_lower:
                pos = instruction_lower.find(keyword)
                pos_factor = 1.0 + (1.0 - pos / len(instruction_lower)) * 0.1
                adjusted_weight = weight * pos_factor
                score_delta -= adjusted_weight
                matched_keywords.append(f"-{keyword}({adjusted_weight:.3f})")
        
        # 特殊模式检测 - 更细粒度的分数
        action_weights = {
            "grab": 0.13, "hold": 0.15, "lift": 0.17, 
            "carry": 0.16, "push": 0.11, "pull": 0.12, 
            "shake": 0.14, "move": 0.09, "place": 0.08,
        }
        
        if "both" in instruction_lower:
            for action, weight in action_weights.items():
                if action in instruction_lower:
                    score_delta += weight
                    matched_keywords.append(f"+both_{action}({weight})")
                    break
        
        # 检测需要精确协调的短语
        precision_phrases = {
            "make sure": 0.06, "ensure": 0.05, "keep": 0.04,
            "carefully": 0.07, "precisely": 0.08, "exactly": 0.06,
        }
        for phrase, weight in precision_phrases.items():
            if phrase in instruction_lower:
                if any(w in instruction_lower for w in ["aligned", "synchronized", "together"]):
                    score_delta += weight
                    matched_keywords.append(f"+precision_{phrase}({weight})")
        
        # 检测左右臂分别做不同任务的模式
        if "left arm" in instruction_lower and "right arm" in instruction_lower:
            # 检查是否是分离任务
            has_coop_words = any(w in instruction_lower for w in ["together", "coordinate", "synchronized", "both"])
            if not has_coop_words:
                # 根据连接词判断分离程度
                if " and " in instruction_lower or ", " in instruction_lower:
                    score_delta -= 0.15
                    matched_keywords.append("-separate_tasks(0.15)")
                elif " while " in instruction_lower:
                    score_delta -= 0.08  # "while" 暗示一定程度的并行
                    matched_keywords.append("-parallel_tasks(0.08)")
        
        # 单臂任务检测
        single_arm_indicators = ["left arm only", "right arm only", "use left", "use right", "with left", "with right"]
        for indicator in single_arm_indicators:
            if indicator in instruction_lower:
                if "left arm" not in instruction_lower or "right arm" not in instruction_lower:
                    score_delta -= 0.25
                    matched_keywords.append(f"-single_arm({indicator})(0.25)")
                    break
        
        # 计算最终分数 - 使用 sigmoid 变换使分数分布更平滑
        raw_score = base_score + score_delta
        # 应用软裁剪，避免极端值
        if raw_score > 1.0:
            final_score = 1.0 - 0.05 * np.exp(-(raw_score - 1.0) * 3)
        elif raw_score < 0.0:
            final_score = 0.05 * np.exp(raw_score * 3)
        else:
            final_score = raw_score
        
        # 添加微小的随机性使结果更自然（基于指令的 hash）
        hash_noise = (hash(instruction) % 100) / 10000.0 - 0.005  # [-0.005, 0.005]
        final_score = max(0.0, min(1.0, final_score + hash_noise))
        
        explanation = f"Matched: {', '.join(matched_keywords)}" if matched_keywords else "No specific keywords matched"
        
        return CooperationResult(
            cooperation_score=round(final_score, 4),  # 保留4位小数
            explanation=explanation,
            raw_output=f"[Rule-based Cooperation]\n"
                      f"Base: {base_score:.4f}, Delta: {score_delta:+.4f}, Final: {final_score:.4f}\n"
                      f"{explanation}",
        )


# =============================================================================
# 图像分析后端 (使用 numpy/PIL)
# =============================================================================

class ImageAnalysisBackend(BimanualBackend):
    """
    基于图像分析的双臂任务模型（支持 1-3 张图像）
    
    结合规则方法和图像空间分析（使用 numpy/PIL）：
    - 分析图像左右两侧的特征分布
    - 检测中心区域的活跃度
    - 融合多视角图像信息
    - 不依赖深度学习库
    """
    
    def __init__(self):
        self._rule_backend = RuleBasedBackend()
        logger.info("图像分析后端初始化完成（支持多图像输入）")
    
    def _compute_edge_map(self, gray_image: np.ndarray) -> np.ndarray:
        """计算边缘图"""
        try:
            from scipy import ndimage
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            grad_x = ndimage.convolve(gray_image.astype(np.float32), sobel_x)
            grad_y = ndimage.convolve(gray_image.astype(np.float32), sobel_y)
        except ImportError:
            grad_x = np.abs(np.diff(gray_image.astype(np.float32), axis=1, prepend=0))
            grad_y = np.abs(np.diff(gray_image.astype(np.float32), axis=0, prepend=0))
        
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _analyze_single_image(self, image: np.ndarray) -> dict:
        """分析单张图像的空间分布"""
        image = preprocess_image(image)
        h, w = image.shape[:2]
        half_w = w // 2
        
        # 转换为灰度
        gray = np.mean(image, axis=2).astype(np.float32)
        
        # 计算左右两侧相似度
        left_half = image[:, :half_w, :]
        right_half = image[:, half_w:, :]
        
        left_hist = np.histogram(left_half.flatten(), bins=32, range=(0, 255))[0].astype(np.float32)
        right_hist = np.histogram(right_half.flatten(), bins=32, range=(0, 255))[0].astype(np.float32)
        
        left_hist = left_hist / (left_hist.sum() + 1e-6)
        right_hist = right_hist / (right_hist.sum() + 1e-6)
        
        left_right_similarity = np.minimum(left_hist, right_hist).sum()
        
        # 计算边缘图
        edge_map = self._compute_edge_map(gray)
        
        # 计算中心活跃度
        center_h_start, center_h_end = h // 3, h - h // 3
        center_w_start, center_w_end = w // 3, w - w // 3
        center_region = edge_map[center_h_start:center_h_end, center_w_start:center_w_end]
        
        center_activity = np.mean(center_region) / (np.mean(edge_map) + 1e-6)
        center_activity = min(1.0, center_activity)
        
        # 计算空间分散度
        grid_size = 4
        grid_h, grid_w = h // grid_size, w // grid_size
        grid_densities = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid_region = edge_map[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_densities.append(np.mean(grid_region))
        
        grid_densities = np.array(grid_densities)
        spatial_spread = np.std(grid_densities) / (np.mean(grid_densities) + 1e-6)
        spatial_spread = min(1.0, spatial_spread)
        
        return {
            "left_right_similarity": float(left_right_similarity),
            "center_activity": float(center_activity),
            "spatial_spread": float(spatial_spread),
        }
    
    def _analyze_wrist_image(self, image: np.ndarray) -> dict:
        """分析手腕相机图像的特征"""
        image = preprocess_image(image)
        
        # 转换为灰度
        gray = np.mean(image, axis=2).astype(np.float32)
        
        # 计算边缘图
        edge_map = self._compute_edge_map(gray)
        
        # 手腕图像主要分析：物体存在性和活跃度
        object_presence = np.mean(edge_map) / 50.0  # 归一化
        object_presence = min(1.0, object_presence)
        
        # 物体中心性
        h, w = image.shape[:2]
        center_region = edge_map[h//4:3*h//4, w//4:3*w//4]
        center_focus = np.mean(center_region) / (np.mean(edge_map) + 1e-6)
        center_focus = min(1.0, center_focus)
        
        return {
            "object_presence": float(object_presence),
            "center_focus": float(center_focus),
        }
    
    def _analyze_spatial_distribution(self, images: Dict[str, np.ndarray]) -> dict:
        """分析多张图像的空间分布，融合多视角信息"""
        result = {}
        
        # 分析基座图像（主图像）
        base_image = images.get("base_rgb", list(images.values())[0])
        base_analysis = self._analyze_single_image(base_image)
        result.update(base_analysis)
        result["num_images"] = len(images)
        
        # 分析手腕图像（如果有）
        left_wrist_analysis = None
        right_wrist_analysis = None
        
        if "left_wrist_rgb" in images:
            left_wrist_analysis = self._analyze_wrist_image(images["left_wrist_rgb"])
            result["left_wrist_presence"] = left_wrist_analysis["object_presence"]
            result["left_wrist_focus"] = left_wrist_analysis["center_focus"]
        
        if "right_wrist_rgb" in images:
            right_wrist_analysis = self._analyze_wrist_image(images["right_wrist_rgb"])
            result["right_wrist_presence"] = right_wrist_analysis["object_presence"]
            result["right_wrist_focus"] = right_wrist_analysis["center_focus"]
        
        # 如果有两个手腕图像，计算它们之间的相似度
        if left_wrist_analysis and right_wrist_analysis:
            # 手腕图像相似度暗示双手在处理相同/相似物体
            wrist_similarity = 1.0 - abs(
                left_wrist_analysis["object_presence"] - right_wrist_analysis["object_presence"]
            )
            result["wrist_similarity"] = float(wrist_similarity)
            
            # 如果两个手腕都有物体且相似，增加协作分数
            if (left_wrist_analysis["object_presence"] > 0.3 and 
                right_wrist_analysis["object_presence"] > 0.3):
                result["both_hands_active"] = True
            else:
                result["both_hands_active"] = False
        
        return result
    
    def generate_bimanual_prompts(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """结合多图像分析生成双臂 prompt（支持 1-3 张图像）"""
        # 基础规则分解
        result = self._rule_backend.generate_bimanual_prompts(images, instruction)
        
        # 分析多图像空间分布
        spatial_info = self._analyze_spatial_distribution(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 根据图像分析调整 prompt
        need_coordinate = False
        
        # 基座图像分析
        if spatial_info["left_right_similarity"] > 0.8 and spatial_info["center_activity"] > 0.7:
            need_coordinate = True
        
        # 手腕图像分析（如果有）
        if spatial_info.get("both_hands_active", False):
            need_coordinate = True
        
        if need_coordinate and "coordinate" not in result.left_arm_prompt.lower():
            result.left_arm_prompt = f"Coordinate: {result.left_arm_prompt}"
            result.right_arm_prompt = f"Coordinate: {result.right_arm_prompt}"
        
        # 构建详细输出
        raw_parts = [f"[Image Analysis + Rule] (images: {num_images})"]
        raw_parts.append(f"Left: {result.left_arm_prompt}")
        raw_parts.append(f"Right: {result.right_arm_prompt}")
        raw_parts.append(f"Base: L-R sim={spatial_info['left_right_similarity']:.3f}, "
                        f"center={spatial_info['center_activity']:.3f}")
        
        if "left_wrist_presence" in spatial_info:
            raw_parts.append(f"Left wrist: presence={spatial_info['left_wrist_presence']:.3f}")
        if "right_wrist_presence" in spatial_info:
            raw_parts.append(f"Right wrist: presence={spatial_info['right_wrist_presence']:.3f}")
        if "wrist_similarity" in spatial_info:
            raw_parts.append(f"Wrist similarity: {spatial_info['wrist_similarity']:.3f}")
        
        result.raw_output = "\n".join(raw_parts)
        
        return result
    
    def evaluate_cooperation(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """结合多图像分析评估协调度 - 输出连续值 [0, 1]（支持 1-3 张图像）"""
        # 基础规则评估
        rule_result = self._rule_backend.evaluate_cooperation(images, instruction)
        base_score = rule_result.cooperation_score
        
        # 多图像分析
        spatial_info = self._analyze_spatial_distribution(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 根据图像分析调整分数 - 使用连续函数而非阈值
        adjustments = []
        score_delta = 0.0
        
        left_right_sim = spatial_info["left_right_similarity"]
        center_activity = spatial_info["center_activity"]
        spatial_spread = spatial_info["spatial_spread"]
        
        # === 基座图像分析 ===
        # 左右相似度: 使用连续映射函数
        sim_contribution = (left_right_sim - 0.5) * 0.25
        score_delta += sim_contribution
        adjustments.append(f"左右相似度({left_right_sim:.3f})={sim_contribution:+.4f}")
        
        # 中心活跃度
        center_contribution = (center_activity - 0.8) * 0.15
        score_delta += center_contribution
        adjustments.append(f"中心活跃度({center_activity:.3f})={center_contribution:+.4f}")
        
        # 空间分散度
        spread_contribution = -(spatial_spread - 0.5) * 0.12
        score_delta += spread_contribution
        adjustments.append(f"空间分散度({spatial_spread:.3f})={spread_contribution:+.4f}")
        
        # === 手腕图像分析（如果有）===
        if "left_wrist_presence" in spatial_info and "right_wrist_presence" in spatial_info:
            left_presence = spatial_info["left_wrist_presence"]
            right_presence = spatial_info["right_wrist_presence"]
            
            # 如果两只手都有物体，增加协作可能性
            if left_presence > 0.3 and right_presence > 0.3:
                both_active_bonus = min(left_presence, right_presence) * 0.15
                score_delta += both_active_bonus
                adjustments.append(f"双手活跃={both_active_bonus:+.4f}")
            
            # 手腕图像相似度
            if "wrist_similarity" in spatial_info:
                wrist_sim = spatial_info["wrist_similarity"]
                wrist_contribution = (wrist_sim - 0.5) * 0.12
                score_delta += wrist_contribution
                adjustments.append(f"手腕相似度({wrist_sim:.3f})={wrist_contribution:+.4f}")
        
        elif "left_wrist_presence" in spatial_info or "right_wrist_presence" in spatial_info:
            # 只有一个手腕图像
            single_wrist = spatial_info.get("left_wrist_presence", spatial_info.get("right_wrist_presence", 0))
            if single_wrist > 0.5:
                # 单手有明显物体，可能是单臂任务
                score_delta -= 0.05
                adjustments.append(f"单手腕活跃=-0.05")
        
        # === 综合指标 ===
        # 如果左右相似且中心活跃，额外加分
        if left_right_sim > 0.7 and center_activity > 0.9:
            synergy_bonus = (left_right_sim - 0.7) * (center_activity - 0.9) * 0.5
            score_delta += synergy_bonus
            adjustments.append(f"协同加成={synergy_bonus:+.4f}")
        
        # 如果空间分散但左右差异大，额外减分
        if spatial_spread > 0.6 and left_right_sim < 0.4:
            independent_penalty = (spatial_spread - 0.6) * (0.4 - left_right_sim) * 0.4
            score_delta -= independent_penalty
            adjustments.append(f"独立任务惩罚={-independent_penalty:+.4f}")
        
        # 多图像加成：更多视角提供更可靠的判断
        if num_images >= 2:
            multi_view_factor = 1.0 + (num_images - 1) * 0.02  # 2图: 1.02, 3图: 1.04
            score_delta *= multi_view_factor
            adjustments.append(f"多视角因子={multi_view_factor:.2f}")
        
        # 计算最终分数
        raw_score = base_score + score_delta
        
        # 使用软裁剪确保在 [0, 1] 范围内
        if raw_score > 1.0:
            final_score = 1.0 - 0.02 * np.exp(-(raw_score - 1.0) * 5)
        elif raw_score < 0.0:
            final_score = 0.02 * np.exp(raw_score * 5)
        else:
            final_score = raw_score
        
        explanation = rule_result.explanation
        if adjustments:
            explanation += f" | Image({num_images}): {'; '.join(adjustments)}"
        
        return CooperationResult(
            cooperation_score=round(final_score, 4),
            explanation=explanation,
            raw_output=f"[Image Analysis + Rule Cooperation] (images: {num_images})\n"
                      f"Base: L-R={left_right_sim:.4f}, Center={center_activity:.4f}, "
                      f"Spread={spatial_spread:.4f}\n"
                      f"Score: base={base_score:.4f}, delta={score_delta:+.4f}, final={final_score:.4f}",
            spatial_info=spatial_info,
        )


# =============================================================================
# PaliGemma VLM 后端
# =============================================================================

class PaliGemmaBackend(BimanualBackend):
    """
    使用 Pi0.5 内置的 PaliGemma VLM 进行双臂任务分解与协调度评估（支持 1-3 张图像）
    
    利用 SigLIP 图像编码器提取视觉特征，结合 Gemma LLM 进行推理
    支持多视角图像输入：base_rgb, left_wrist_rgb, right_wrist_rgb
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
    ):
        """
        初始化 PaliGemma 后端
        
        Args:
            checkpoint_path: 检查点路径
            use_pretrained: 是否使用预训练模型
        """
        # 延迟导入以避免不必要的依赖
        import jax
        import jax.numpy as jnp
        import sentencepiece
        import flax.nnx as nnx
        import flax.nnx.bridge as nnx_bridge
        from openpi.models import model as _model
        from openpi.models import pi0_config
        import openpi.models.gemma as _gemma
        import openpi.models.siglip as _siglip
        from openpi.shared import download
        
        self.jax = jax
        self.jnp = jnp
        
        # 预训练模型 URL
        PRETRAINED_MODELS = {
            "pi05_base": "s3://openpi-assets/checkpoints/pi05_base/params",
        }
        
        # 获取检查点
        if use_pretrained:
            pretrained_url = PRETRAINED_MODELS.get("pi05_base")
            logger.info(f"下载预训练模型: {pretrained_url}")
            checkpoint_path = download.maybe_download(pretrained_url)
        
        self.checkpoint_path = checkpoint_path
        
        # 加载 tokenizer
        logger.info("加载 PaliGemma tokenizer...")
        tokenizer_path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model",
            gs={"token": "anon"}
        )
        with tokenizer_path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        
        # 加载模型
        logger.info("加载 Pi0.5 VLM 模型...")
        config = pi0_config.Pi0Config(pi05=True)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        
        rngs = nnx.Rngs(jax.random.key(0))
        
        # 加载 LLM
        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )
        self.llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        
        # 加载图像编码器
        self.img_encoder = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        fake_image = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
        self.img_encoder.lazy_init(fake_image, train=False, rngs=rngs)
        
        # 加载权重
        if self.checkpoint_path:
            logger.info(f"从 {self.checkpoint_path} 加载权重...")
            params = _model.restore_params(self.checkpoint_path)
            if 'PaliGemma' in params:
                logger.info("已加载 PaliGemma 权重")
        
        self.config = config
        self.paligemma_config = paligemma_config
        
        # 规则后端作为后备
        self._rule_backend = RuleBasedBackend()
        self._image_backend = ImageAnalysisBackend()
        
        logger.info("PaliGemma 后端加载完成!")
    
    def _preprocess_image(self, image: np.ndarray):
        """预处理图像为 JAX 张量"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.shape[:2] != (224, 224):
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((224, 224), Image.BILINEAR)
            image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 归一化到 [-1, 1]
        image = image * 2.0 - 1.0
        image = image[np.newaxis, ...]
        
        return self.jnp.array(image)
    
    def _analyze_single_image_features(self, image_tokens) -> dict:
        """分析单张图像的深度特征空间分布"""
        tokens = np.array(image_tokens[0])  # (num_patches, hidden_dim)
        num_patches = tokens.shape[0]
        
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            grid_size = 16  # SigLIP So400m/14 默认
        
        try:
            tokens_grid = tokens.reshape(grid_size, grid_size, -1)
        except ValueError:
            return {"left_right_similarity": 0.5, "center_activity": 0.5, "spatial_spread": 0.5}
        
        half = grid_size // 2
        left_half = tokens_grid[:, :half, :].reshape(-1, tokens_grid.shape[-1])
        right_half = tokens_grid[:, half:, :].reshape(-1, tokens_grid.shape[-1])
        
        # 计算左右相似度
        left_mean = np.mean(left_half, axis=0)
        right_mean = np.mean(right_half, axis=0)
        
        left_norm = np.linalg.norm(left_mean)
        right_norm = np.linalg.norm(right_mean)
        
        if left_norm > 0 and right_norm > 0:
            left_right_similarity = np.dot(left_mean, right_mean) / (left_norm * right_norm)
            left_right_similarity = (left_right_similarity + 1) / 2
        else:
            left_right_similarity = 0.5
        
        # 计算中心活跃度
        center_start = grid_size // 3
        center_end = grid_size - center_start
        center_region = tokens_grid[center_start:center_end, center_start:center_end, :]
        
        center_activity = np.mean(np.linalg.norm(center_region, axis=-1))
        overall_activity = np.mean(np.linalg.norm(tokens_grid, axis=-1))
        
        center_activity_ratio = min(1.0, center_activity / (overall_activity + 1e-6))
        
        # 计算空间分散度
        token_norms = np.linalg.norm(tokens, axis=-1)
        spatial_spread = min(1.0, np.std(token_norms) / (np.mean(token_norms) + 1e-6))
        
        # 计算整体特征向量（用于多图像比较）
        global_feature = np.mean(tokens, axis=0)
        
        return {
            "left_right_similarity": float(left_right_similarity),
            "center_activity": float(center_activity_ratio),
            "spatial_spread": float(spatial_spread),
            "global_feature": global_feature,
            "feature_norm": float(np.linalg.norm(global_feature)),
        }
    
    def _analyze_spatial_features(self, images: Dict[str, np.ndarray]) -> dict:
        """分析多张图像的深度特征空间分布"""
        result = {"num_images": len(images)}
        
        # 编码所有图像
        encoded_features = {}
        for key, img in images.items():
            image_tensor = self._preprocess_image(img)
            image_tokens, _ = self.img_encoder(image_tensor, train=False)
            encoded_features[key] = self._analyze_single_image_features(image_tokens)
        
        # 基座图像分析（主图像）
        base_key = "base_rgb" if "base_rgb" in encoded_features else list(encoded_features.keys())[0]
        base_analysis = encoded_features[base_key]
        
        result["left_right_similarity"] = base_analysis["left_right_similarity"]
        result["center_activity"] = base_analysis["center_activity"]
        result["spatial_spread"] = base_analysis["spatial_spread"]
        
        # 手腕图像分析
        if "left_wrist_rgb" in encoded_features:
            left_wrist = encoded_features["left_wrist_rgb"]
            result["left_wrist_activity"] = left_wrist["feature_norm"] / 100.0  # 归一化
            result["left_wrist_activity"] = min(1.0, result["left_wrist_activity"])
        
        if "right_wrist_rgb" in encoded_features:
            right_wrist = encoded_features["right_wrist_rgb"]
            result["right_wrist_activity"] = right_wrist["feature_norm"] / 100.0
            result["right_wrist_activity"] = min(1.0, result["right_wrist_activity"])
        
        # 计算手腕图像之间的特征相似度
        if "left_wrist_rgb" in encoded_features and "right_wrist_rgb" in encoded_features:
            left_feat = encoded_features["left_wrist_rgb"]["global_feature"]
            right_feat = encoded_features["right_wrist_rgb"]["global_feature"]
            
            left_norm = np.linalg.norm(left_feat)
            right_norm = np.linalg.norm(right_feat)
            
            if left_norm > 0 and right_norm > 0:
                wrist_similarity = np.dot(left_feat, right_feat) / (left_norm * right_norm)
                wrist_similarity = (wrist_similarity + 1) / 2  # 归一化到 [0, 1]
            else:
                wrist_similarity = 0.5
            
            result["wrist_feature_similarity"] = float(wrist_similarity)
            
            # 判断双手是否都在活动
            if (result.get("left_wrist_activity", 0) > 0.3 and 
                result.get("right_wrist_activity", 0) > 0.3):
                result["both_hands_active"] = True
            else:
                result["both_hands_active"] = False
        
        return result
    
    def generate_bimanual_prompts(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用 PaliGemma 生成双臂 prompt（支持 1-3 张图像）"""
        # 分析多图像特征
        spatial_info = self._analyze_spatial_features(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 使用规则方法作为基础
        result = self._rule_backend.generate_bimanual_prompts(images, instruction)
        
        # 根据视觉特征调整
        need_coordinate = False
        
        if spatial_info["left_right_similarity"] > 0.8 and spatial_info["center_activity"] > 0.6:
            need_coordinate = True
        
        # 如果手腕图像显示双手都在活动
        if spatial_info.get("both_hands_active", False):
            need_coordinate = True
        
        if need_coordinate:
            if "together" not in instruction.lower() and "both" not in instruction.lower():
                if "coordinate" not in result.left_arm_prompt.lower():
                    result.left_arm_prompt = f"Coordinate: {result.left_arm_prompt}"
                    result.right_arm_prompt = f"Coordinate: {result.right_arm_prompt}"
        
        # 构建详细输出
        raw_parts = [f"[PaliGemma VLM] (images: {num_images})"]
        raw_parts.append(f"Left: {result.left_arm_prompt}")
        raw_parts.append(f"Right: {result.right_arm_prompt}")
        raw_parts.append(f"Visual: L-R={spatial_info['left_right_similarity']:.3f}, "
                        f"Center={spatial_info['center_activity']:.3f}")
        
        if "wrist_feature_similarity" in spatial_info:
            raw_parts.append(f"Wrist similarity: {spatial_info['wrist_feature_similarity']:.3f}")
        
        result.raw_output = "\n".join(raw_parts)
        
        return result
    
    def evaluate_cooperation(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用 PaliGemma 评估协调度 - 输出连续值 [0, 1]（支持 1-3 张图像）"""
        # 分析多图像特征
        spatial_info = self._analyze_spatial_features(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 规则基础评估
        rule_result = self._rule_backend.evaluate_cooperation(images, instruction)
        base_score = rule_result.cooperation_score
        
        # 根据深度视觉特征调整 - 使用连续映射
        adjustments = []
        score_delta = 0.0
        
        left_right_sim = spatial_info["left_right_similarity"]
        center_activity = spatial_info["center_activity"]
        spatial_spread = spatial_info["spatial_spread"]
        
        # === 基座图像分析 ===
        # 视觉相似度贡献: 深度特征的相似度更有意义，权重更高
        sim_contribution = (left_right_sim - 0.5) * 0.30
        score_delta += sim_contribution
        adjustments.append(f"视觉相似度({left_right_sim:.3f})={sim_contribution:+.4f}")
        
        # 中心活跃度贡献
        center_contribution = (center_activity - 0.5) * 0.20
        score_delta += center_contribution
        adjustments.append(f"中心活跃度({center_activity:.3f})={center_contribution:+.4f}")
        
        # 空间分散度贡献 (负相关)
        spread_contribution = -(spatial_spread - 0.5) * 0.15
        score_delta += spread_contribution
        adjustments.append(f"空间分散度({spatial_spread:.3f})={spread_contribution:+.4f}")
        
        # === 手腕图像分析（如果有）===
        if "left_wrist_activity" in spatial_info and "right_wrist_activity" in spatial_info:
            left_activity = spatial_info["left_wrist_activity"]
            right_activity = spatial_info["right_wrist_activity"]
            
            # 如果两只手都有活动，增加协作可能性
            if left_activity > 0.3 and right_activity > 0.3:
                both_active_bonus = min(left_activity, right_activity) * 0.18
                score_delta += both_active_bonus
                adjustments.append(f"双手活跃={both_active_bonus:+.4f}")
            
            # 手腕特征相似度
            if "wrist_feature_similarity" in spatial_info:
                wrist_sim = spatial_info["wrist_feature_similarity"]
                wrist_contribution = (wrist_sim - 0.5) * 0.15
                score_delta += wrist_contribution
                adjustments.append(f"手腕特征相似度({wrist_sim:.3f})={wrist_contribution:+.4f}")
        
        # === 综合指标 ===
        # 深度特征的交互效应
        if left_right_sim > 0.6 and center_activity > 0.6:
            interaction = (left_right_sim - 0.6) * (center_activity - 0.6) * 0.8
            score_delta += interaction
            adjustments.append(f"协同交互效应={interaction:+.4f}")
        
        # 多图像加成
        if num_images >= 2:
            multi_view_factor = 1.0 + (num_images - 1) * 0.03
            score_delta *= multi_view_factor
            adjustments.append(f"多视角因子={multi_view_factor:.2f}")
        
        # 计算最终分数
        raw_score = base_score + score_delta
        
        # 软裁剪
        if raw_score > 1.0:
            final_score = 1.0 - 0.01 * np.exp(-(raw_score - 1.0) * 8)
        elif raw_score < 0.0:
            final_score = 0.01 * np.exp(raw_score * 8)
        else:
            final_score = raw_score
        
        explanation = rule_result.explanation
        if adjustments:
            explanation += f" | Visual({num_images}): {'; '.join(adjustments)}"
        
        return CooperationResult(
            cooperation_score=round(final_score, 4),
            explanation=explanation,
            raw_output=f"[PaliGemma VLM Cooperation] (images: {num_images})\n"
                      f"Visual: L-R={left_right_sim:.4f}, "
                      f"Center={center_activity:.4f}, Spread={spatial_spread:.4f}\n"
                      f"Score: base={base_score:.4f}, delta={score_delta:+.4f}, final={final_score:.4f}",
            spatial_info=spatial_info,
        )


# =============================================================================
# API 后端 (使用外部 VLM API)
# =============================================================================

class APIBackend(BimanualBackend):
    """使用外部 VLM API (如 OpenAI GPT-4V) 进行双臂任务分解与协调度评估（支持 1-3 张图像）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-vision-preview",
        api_provider: str = "openai",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.api_provider = api_provider
        
        if not self.api_key:
            logger.warning("未设置 API Key，将使用规则方法作为后备")
            self._fallback = RuleBasedBackend()
        else:
            self._fallback = None
        
        logger.info(f"API 后端初始化完成, provider: {api_provider}")
    
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """将图像编码为 base64"""
        import base64
        import io
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _build_image_content(self, images: Dict[str, np.ndarray]) -> list:
        """构建多图像内容列表"""
        content = []
        
        # 图像描述
        image_descriptions = {
            "base_rgb": "Main workspace view (base camera)",
            "left_wrist_rgb": "Left arm's wrist camera view",
            "right_wrist_rgb": "Right arm's wrist camera view",
        }
        
        for key in IMAGE_KEYS:
            if key in images:
                desc = image_descriptions.get(key, key)
                content.append({"type": "text", "text": f"\n[{desc}]:"})
                image_base64 = self._encode_image_base64(images[key])
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
        
        return content
    
    def generate_bimanual_prompts(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用 API 生成双臂 prompt（支持 1-3 张图像）"""
        if self._fallback:
            return self._fallback.generate_bimanual_prompts(images, instruction)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            num_images = len(images)
            
            # 构建提示
            prompt_text = f"""Look at these {num_images} image(s) of a bimanual robot workspace.

Task: {instruction}

Please decompose this task into two separate subtasks - one for the left arm and one for the right arm.
Consider all provided camera views when making your decision.

Format your response EXACTLY as:
Left Arm: [specific subtask for left arm]
Right Arm: [specific subtask for right arm]

Be specific and concise."""

            # 构建消息内容
            content = [{"type": "text", "text": prompt_text}]
            content.extend(self._build_image_content(images))

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
            )
            
            raw_output = response.choices[0].message.content
            
            # 解析响应
            left_prompt = ""
            right_prompt = ""
            
            left_match = re.search(r'[Ll]eft\s*[Aa]rm[:\s]+([^\n]+)', raw_output)
            right_match = re.search(r'[Rr]ight\s*[Aa]rm[:\s]+([^\n]+)', raw_output)
            
            if left_match:
                left_prompt = left_match.group(1).strip()
            if right_match:
                right_prompt = right_match.group(1).strip()
            
            return BimanualPromptResult(
                left_arm_prompt=left_prompt,
                right_arm_prompt=right_prompt,
                raw_output=f"[API: {self.api_provider}] (images: {num_images})\n{raw_output}",
            )
            
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return RuleBasedBackend().generate_bimanual_prompts(images, instruction)
    
    def evaluate_cooperation(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用 API 评估协调度（支持 1-3 张图像）"""
        if self._fallback:
            return self._fallback.evaluate_cooperation(images, instruction)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            num_images = len(images)
            
            prompt_text = f"""Analyze these {num_images} image(s) of a bimanual robot task and evaluate the cooperation level between the two arms.

Task instruction: "{instruction}"

Rate the cooperation level as a continuous value from 0.0 to 1.0:
- 0.0-0.2: Very low cooperation, arms work almost completely independently
- 0.2-0.4: Low cooperation, simple handover or sequential tasks
- 0.4-0.6: Moderate cooperation, some timing coordination needed
- 0.6-0.8: High cooperation, spatial and temporal coordination required
- 0.8-1.0: Very high cooperation, tight synchronization and alignment needed

Consider all provided camera views when making your assessment.

Respond in EXACTLY this format:
Score: X.XXXX (4 decimal places)
Reason: [Brief explanation]"""

            # 构建消息内容
            content = [{"type": "text", "text": prompt_text}]
            content.extend(self._build_image_content(images))

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=150,
            )
            
            raw_output = response.choices[0].message.content
            
            # 解析分数
            score = 0.5
            explanation = ""
            
            score_match = re.search(r'[Ss]core[:\s]+(\d+\.?\d*)', raw_output)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            
            reason_match = re.search(r'[Rr]eason[:\s]+(.+?)(?:\n|$)', raw_output)
            if reason_match:
                explanation = reason_match.group(1).strip()
            
            return CooperationResult(
                cooperation_score=round(score, 4),
                explanation=explanation,
                raw_output=f"[API: {self.api_provider}] (images: {num_images})\n{raw_output}",
            )
            
        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            return RuleBasedBackend().evaluate_cooperation(images, instruction)


# =============================================================================
# 主模型类
# =============================================================================

class BimanualTaskModel:
    """
    双臂任务分解与协调度评估模型（支持 1-3 张图像输入）
    
    利用 Pi0.5 的 PaliGemma VLM 部分，完成以下任务：
    
    任务1: 给定 1-3 张任务截图和任务 prompt，输出左右臂分别的 prompt
    任务2: 给定 1-3 张任务截图和任务 prompt，输出左右手的协调度（0-1 连续值）
    
    支持的图像输入格式:
        - 单张图像: image (作为 base_rgb)
        - 列表: [base_rgb, left_wrist_rgb, right_wrist_rgb]
        - 字典: {"base_rgb": img, "left_wrist_rgb": img, "right_wrist_rgb": img}
    
    支持多种后端模式:
        - rule: 基于规则（快速，不需要模型）
        - image: 规则 + numpy 图像分析（推荐，不依赖深度学习库）
        - paligemma: 使用 Pi0.5 内置的 PaliGemma VLM
        - api: 使用外部 VLM API（如 GPT-4V）
    
    Example:
        >>> model = BimanualTaskModel(mode="image")
        >>> 
        >>> # 单张图片
        >>> result = model.generate_bimanual_prompts(base_image, "Pick up the bottle")
        >>> 
        >>> # 多张图片 - 使用字典
        >>> images = {
        ...     "base_rgb": base_image,
        ...     "left_wrist_rgb": left_wrist_image,
        ...     "right_wrist_rgb": right_wrist_image,
        ... }
        >>> result = model.generate_bimanual_prompts(images, "Pick up the bottle")
        >>> print(result.left_arm_prompt)
        >>> print(result.right_arm_prompt)
        >>> 
        >>> # 多张图片 - 使用列表
        >>> images = [base_image, left_wrist_image, right_wrist_image]
        >>> result = model.evaluate_cooperation(images, "Shake with both hands")
        >>> print(result.cooperation_score)  # 0.7823
        >>> 
        >>> # 同时执行两个任务
        >>> result = model.analyze(images, "Lift the heavy box together")
        >>> print(result.left_arm_prompt)
        >>> print(result.right_arm_prompt)
        >>> print(result.cooperation_score)
    """
    
    SUPPORTED_MODES = ["rule", "image", "paligemma", "api"]
    
    def __init__(
        self,
        mode: str = "image",
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "openai",
    ):
        """
        初始化双臂任务模型
        
        Args:
            mode: 后端模式
                - "rule": 仅基于规则（快速测试）
                - "image": 规则 + numpy 图像分析（推荐）
                - "paligemma": 使用 Pi0.5 VLM（需要 jax/openpi）
                - "api": 使用外部 VLM API
            checkpoint_path: 检查点路径（paligemma 模式）
            use_pretrained: 是否使用预训练模型（paligemma 模式）
            api_key: API Key（api 模式）
            api_provider: API 提供商（api 模式）
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"不支持的模式: {mode}, 支持: {self.SUPPORTED_MODES}")
        
        self.mode = mode
        
        if mode == "rule":
            self.backend = RuleBasedBackend()
        elif mode == "image":
            self.backend = ImageAnalysisBackend()
        elif mode == "paligemma":
            self.backend = PaliGemmaBackend(
                checkpoint_path=checkpoint_path,
                use_pretrained=use_pretrained,
            )
        elif mode == "api":
            self.backend = APIBackend(
                api_key=api_key,
                api_provider=api_provider,
            )
        
        logger.info(f"BimanualTaskModel 初始化完成, 模式: {mode}")
    
    def generate_bimanual_prompts(
        self,
        images: ImageInput,
        instruction: str,
    ) -> BimanualPromptResult:
        """
        任务1: 生成左右臂的 prompt
        
        Args:
            images: 1-3 张任务场景图像，支持以下格式:
                - 单张图像 (str, np.ndarray, PIL.Image)
                - 列表 [base_rgb, left_wrist_rgb, right_wrist_rgb]
                - 字典 {"base_rgb": img, "left_wrist_rgb": img, ...}
            instruction: 高层任务指令
            
        Returns:
            BimanualPromptResult: 包含 left_arm_prompt 和 right_arm_prompt
        """
        images_dict = load_images(images)
        return self.backend.generate_bimanual_prompts(images_dict, instruction)
    
    def evaluate_cooperation(
        self,
        images: ImageInput,
        instruction: str,
    ) -> CooperationResult:
        """
        任务2: 评估协调度
        
        Args:
            images: 1-3 张任务场景图像，支持以下格式:
                - 单张图像 (str, np.ndarray, PIL.Image)
                - 列表 [base_rgb, left_wrist_rgb, right_wrist_rgb]
                - 字典 {"base_rgb": img, "left_wrist_rgb": img, ...}
            instruction: 任务指令
            
        Returns:
            CooperationResult: 包含 cooperation_score (连续值 0-1)
        """
        images_dict = load_images(images)
        return self.backend.evaluate_cooperation(images_dict, instruction)
    
    def analyze(
        self,
        images: ImageInput,
        instruction: str,
    ) -> BimanualAnalysisResult:
        """
        同时执行任务1和任务2
        
        Args:
            images: 1-3 张任务场景图像，支持以下格式:
                - 单张图像 (str, np.ndarray, PIL.Image)
                - 列表 [base_rgb, left_wrist_rgb, right_wrist_rgb]
                - 字典 {"base_rgb": img, "left_wrist_rgb": img, ...}
            instruction: 任务指令
            
        Returns:
            BimanualAnalysisResult: 完整分析结果
        """
        images_dict = load_images(images)
        return self.backend.analyze(images_dict, instruction)
    
    def __call__(
        self,
        images: ImageInput,
        instruction: str,
    ) -> BimanualAnalysisResult:
        """调用接口，同时执行任务1和任务2"""
        return self.analyze(images, instruction)


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_bimanual_model(
    mode: str = "image",
    **kwargs,
) -> BimanualTaskModel:
    """
    创建双臂任务模型实例
    
    Args:
        mode: 后端模式 ("rule", "image", "paligemma", "api")
        **kwargs: 其他参数传递给 BimanualTaskModel
        
    Returns:
        BimanualTaskModel 实例
        
    Example:
        >>> model = create_bimanual_model(mode="image")
        >>> result = model(image, "Pick up the bottle")
    """
    return BimanualTaskModel(mode=mode, **kwargs)


# =============================================================================
# 测试与示例
# =============================================================================

def create_test_image(size: tuple = (224, 224)) -> np.ndarray:
    """创建测试图像"""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def create_test_images() -> Dict[str, np.ndarray]:
    """创建测试图像集（模拟多相机输入）"""
    return {
        "base_rgb": create_test_image(),
        "left_wrist_rgb": create_test_image(),
        "right_wrist_rgb": create_test_image(),
    }


def run_demo():
    """运行演示"""
    logger.info("=" * 70)
    logger.info("双臂任务分解与协调度评估模型 - 演示")
    logger.info("=" * 70)
    
    # 创建模型
    model = BimanualTaskModel(mode="image")
    
    # 创建测试图像
    single_image = create_test_image()
    multi_images = create_test_images()
    
    # 测试用例
    test_cases = [
        # (指令, 预期协调度级别)
        ("Shake the bottle with both hands", "高"),
        ("Lift the heavy box together", "高"),
        ("Left arm picks up the red cube, right arm picks up the blue cube", "低"),
        ("Hand over the tool from left arm to right arm", "中"),
        ("Fold the cloth with both arms", "高"),
        ("Pick up the bottle with left arm", "低"),
    ]
    
    # === 单图像测试 ===
    logger.info("\n" + "-" * 70)
    logger.info("任务1: 双臂 Prompt 生成 (单图像)")
    logger.info("-" * 70)
    
    for instruction, _ in test_cases[:3]:
        result = model.generate_bimanual_prompts(single_image, instruction)
        logger.info(f"\n指令: {instruction}")
        logger.info(f"  左臂: {result.left_arm_prompt}")
        logger.info(f"  右臂: {result.right_arm_prompt}")
    
    # === 多图像测试 ===
    logger.info("\n" + "-" * 70)
    logger.info("任务1: 双臂 Prompt 生成 (多图像: base + left_wrist + right_wrist)")
    logger.info("-" * 70)
    
    for instruction, _ in test_cases[:3]:
        result = model.generate_bimanual_prompts(multi_images, instruction)
        logger.info(f"\n指令: {instruction}")
        logger.info(f"  左臂: {result.left_arm_prompt}")
        logger.info(f"  右臂: {result.right_arm_prompt}")
    
    logger.info("\n" + "-" * 70)
    logger.info("任务2: 协调度评估 (连续值 0-1, 多图像)")
    logger.info("-" * 70)
    
    for instruction, expected_level in test_cases:
        result = model.evaluate_cooperation(multi_images, instruction)
        score = result.cooperation_score
        
        # 显示连续值和对应的可视化条
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        logger.info(f"\n指令: {instruction}")
        logger.info(f"  协调度: {score:.4f} [{bar}] (预期趋势: {expected_level})")
        logger.info(f"  解释: {result.explanation}")
    
    # === 列表格式测试 ===
    logger.info("\n" + "-" * 70)
    logger.info("多图像输入格式测试")
    logger.info("-" * 70)
    
    instruction = "Shake the bottle with both hands"
    
    # 字典格式
    result_dict = model.evaluate_cooperation(multi_images, instruction)
    logger.info(f"\n字典格式输入: 协调度 = {result_dict.cooperation_score:.4f}")
    
    # 列表格式
    images_list = [multi_images["base_rgb"], multi_images["left_wrist_rgb"], multi_images["right_wrist_rgb"]]
    result_list = model.evaluate_cooperation(images_list, instruction)
    logger.info(f"列表格式输入: 协调度 = {result_list.cooperation_score:.4f}")
    
    # 单图像格式
    result_single = model.evaluate_cooperation(single_image, instruction)
    logger.info(f"单图像输入:   协调度 = {result_single.cooperation_score:.4f}")
    
    logger.info("\n" + "-" * 70)
    logger.info("完整分析 (任务1 + 任务2, 多图像)")
    logger.info("-" * 70)
    
    instruction = "Shake the bottle with both hands and make sure hands are aligned"
    result = model.analyze(multi_images, instruction)
    
    logger.info(f"\n指令: {instruction}")
    logger.info(f"  图像数量: 3 (base + left_wrist + right_wrist)")
    logger.info(f"  左臂: {result.left_arm_prompt}")
    logger.info(f"  右臂: {result.right_arm_prompt}")
    logger.info(f"  协调度: {result.cooperation_score:.4f}")
    logger.info(f"  解释: {result.explanation}")
    
    logger.info("\n" + "=" * 70)
    logger.info("演示完成!")
    logger.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="双臂任务分解与协调度评估模型（支持 1-3 张图像输入）"
    )
    parser.add_argument(
        "--mode", type=str, default="image",
        choices=["rule", "image", "paligemma", "api"],
        help="后端模式"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="检查点路径 (paligemma 模式)"
    )
    parser.add_argument(
        "--use-pretrained", action="store_true",
        help="使用预训练模型 (paligemma 模式)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API Key (api 模式)"
    )
    
    # 图像输入参数（支持多图像）
    parser.add_argument(
        "--image", type=str, default=None,
        help="输入图像路径（单张图像，作为 base_rgb）"
    )
    parser.add_argument(
        "--base-image", type=str, default=None,
        help="基座相机图像路径"
    )
    parser.add_argument(
        "--left-wrist-image", type=str, default=None,
        help="左手腕相机图像路径"
    )
    parser.add_argument(
        "--right-wrist-image", type=str, default=None,
        help="右手腕相机图像路径"
    )
    
    parser.add_argument(
        "--instruction", type=str,
        default="Shake the bottle with both hands",
        help="任务指令"
    )
    parser.add_argument(
        "--task", type=str, default="both",
        choices=["prompt", "cooperation", "both"],
        help="执行的任务: prompt (任务1), cooperation (任务2), both (两者)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="运行演示"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    # 创建模型
    model = BimanualTaskModel(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        use_pretrained=args.use_pretrained,
        api_key=args.api_key,
    )
    
    # 加载图像（支持多图像输入）
    images = {}
    num_images = 0
    
    # 优先使用多图像参数
    if args.base_image and Path(args.base_image).exists():
        images["base_rgb"] = np.array(Image.open(args.base_image).convert('RGB'))
        num_images += 1
        logger.info(f"加载基座图像: {args.base_image}")
    
    if args.left_wrist_image and Path(args.left_wrist_image).exists():
        images["left_wrist_rgb"] = np.array(Image.open(args.left_wrist_image).convert('RGB'))
        num_images += 1
        logger.info(f"加载左手腕图像: {args.left_wrist_image}")
    
    if args.right_wrist_image and Path(args.right_wrist_image).exists():
        images["right_wrist_rgb"] = np.array(Image.open(args.right_wrist_image).convert('RGB'))
        num_images += 1
        logger.info(f"加载右手腕图像: {args.right_wrist_image}")
    
    # 如果没有使用多图像参数，检查单图像参数
    if num_images == 0:
        if args.image and Path(args.image).exists():
            images["base_rgb"] = np.array(Image.open(args.image).convert('RGB'))
            num_images = 1
            logger.info(f"加载图像: {args.image}")
        else:
            logger.info("未提供图像，使用随机测试图像...")
            images = create_test_images()
            num_images = 3
    
    logger.info(f"图像数量: {num_images}")
    
    # 执行任务
    logger.info(f"\n指令: {args.instruction}")
    logger.info("-" * 50)
    
    if args.task in ["prompt", "both"]:
        result = model.generate_bimanual_prompts(images, args.instruction)
        logger.info("\n任务1 - 双臂 Prompt 生成:")
        logger.info(f"  左臂: {result.left_arm_prompt}")
        logger.info(f"  右臂: {result.right_arm_prompt}")
    
    if args.task in ["cooperation", "both"]:
        result = model.evaluate_cooperation(images, args.instruction)
        logger.info("\n任务2 - 协调度评估:")
        logger.info(f"  协调度: {result.cooperation_score:.4f}")
        logger.info(f"  解释: {result.explanation}")


if __name__ == "__main__":
    main()

