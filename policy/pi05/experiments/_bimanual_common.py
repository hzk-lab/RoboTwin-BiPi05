#!/usr/bin/env python3
"""
双臂任务模型共享模块

提供任务分解和协调度评估共用的工具：
- 数据类定义
- 图像加载/预处理
- Inference Time Benchmark
- 常量定义
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, List

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义
# =============================================================================

# 类型别名
ImageType = Union[str, np.ndarray, Image.Image]
ImageInput = Union[ImageType, List[ImageType], Dict[str, ImageType]]

# 图像键名常量
IMAGE_KEYS = ["base_rgb", "left_wrist_rgb", "right_wrist_rgb"]

# 预训练模型 URL
PRETRAINED_MODELS = {
    "pi05_base": "s3://openpi-assets/checkpoints/pi05_base/params",
    "pi0_base": "s3://openpi-assets/checkpoints/pi0_base/params",
}


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
        # 字典输入
        for key in IMAGE_KEYS:
            if key in images and images[key] is not None:
                result[key] = load_single_image(images[key])
        
        if not result:
            raise ValueError("至少需要提供一张图像")
        
        if "base_rgb" not in result:
            first_key = list(result.keys())[0]
            result["base_rgb"] = result[first_key]
            
    elif isinstance(images, (list, tuple)):
        # 列表输入
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
    """兼容旧接口：加载单张图像，如果输入多张图像返回 base_rgb"""
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
        pil_image = pil_image.resize(target_size[::-1], Image.BILINEAR)
        image = np.array(pil_image)
    
    return image


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


# =============================================================================
# Inference Time Benchmark
# =============================================================================

def benchmark_inference(
    model,
    images: ImageInput,
    instruction: str,
    num_warmup: int = 3,
    num_runs: int = 10,
    verbose: bool = True,
) -> dict:
    """
    标准 inference time 测试
    
    测量的是模型已加载后，处理单个样本的端到端时间，包括：
    - 输入预处理（图像加载、resize、normalize 等）
    - 模型前向传播
    - 输出后处理（解码、解析等）
    
    不包括：
    - 模型加载时间
    - 权重下载时间
    - JIT 首次编译时间（通过 warmup 排除）
    
    Args:
        model: 已加载好的模型实例
        images: 测试图像（支持多图像输入）
        instruction: 测试指令
        num_warmup: 预热次数（排除 JIT 编译等一次性开销）
        num_runs: 正式测试次数
        verbose: 是否打印详细信息
        
    Returns:
        dict: benchmark 结果
    """
    # ========== 预热阶段（不计时）==========
    if verbose:
        logger.info(f"Warming up ({num_warmup} runs)...")
    
    for i in range(num_warmup):
        _ = model(images, instruction)
        if verbose:
            logger.info(f"  Warmup {i+1}/{num_warmup} done")
    
    # ========== 正式测试（计时）==========
    if verbose:
        logger.info(f"Benchmarking ({num_runs} runs)...")
    
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        result = model(images, instruction)
        end = time.perf_counter()
        
        elapsed = end - start
        times.append(elapsed)
        
        if verbose:
            logger.info(f"  Run {i+1}/{num_runs}: {elapsed * 1000:.2f} ms")
    
    times = np.array(times)
    
    results = {
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
        "median_ms": float(np.median(times) * 1000),
        "all_times_ms": (times * 1000).tolist(),
        "num_runs": num_runs,
        "num_warmup": num_warmup,
    }
    
    return results


def print_benchmark_results(results: dict, model_name: str = "Model"):
    """打印 benchmark 结果"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Inference Time Benchmark Results - {model_name}")
    logger.info("=" * 60)
    logger.info(f"  Warmup runs:  {results['num_warmup']}")
    logger.info(f"  Test runs:    {results['num_runs']}")
    logger.info("-" * 60)
    logger.info(f"  Mean:         {results['mean_ms']:.2f} ms")
    logger.info(f"  Std:          {results['std_ms']:.2f} ms")
    logger.info(f"  Min:          {results['min_ms']:.2f} ms")
    logger.info(f"  Max:          {results['max_ms']:.2f} ms")
    logger.info(f"  Median:       {results['median_ms']:.2f} ms")
    logger.info("-" * 60)
    logger.info(f"  Throughput:   {1000 / results['mean_ms']:.2f} samples/sec")
    logger.info("=" * 60)

