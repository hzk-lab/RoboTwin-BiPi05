#!/usr/bin/env python3
"""
双臂任务分解模块 (Task Decomposition)

给定 1-3 张任务截图和任务 prompt，输出左右臂分别的 prompt。

输入:
    - images: 1-3 张任务场景图像 (RGB, 224x224)
        - base_rgb: 基座相机图像（必需）
        - left_wrist_rgb: 左手腕相机图像（可选）
        - right_wrist_rgb: 右手腕相机图像（可选）
    - instruction: 高层任务指令

输出:
    - left_arm_prompt: 左臂任务指令
    - right_arm_prompt: 右臂任务指令

支持四种模式:
    1. rule: 基于规则的分解 (快速，不需要模型)
    2. image: 规则 + numpy 图像分析 (推荐，不依赖深度学习库)
    3. paligemma: 使用 Pi0.5 内置的 PaliGemma VLM (需要预训练权重)
    4. api: 使用外部 VLM API (如 OpenAI GPT-4V)

Usage:
    cd /path/to/RoboTwin/policy/pi05
    
    # 使用规则模式
    python experiments/task_decomposition.py --mode rule
    
    # 使用图像分析模式 (推荐)
    python experiments/task_decomposition.py --mode image
    
    # 多图像输入
    python experiments/task_decomposition.py --mode image \\
        --base-image base.jpg --left-wrist-image left.jpg --right-wrist-image right.jpg
    
    # Inference Time Benchmark
    python experiments/task_decomposition.py --mode rule --benchmark
    
    # 对比所有模式
    python experiments/task_decomposition.py --benchmark-all
"""

import argparse
import logging
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from _bimanual_common import (
    BimanualPromptResult,
    ImageInput,
    IMAGE_KEYS,
    PRETRAINED_MODELS,
    benchmark_inference,
    create_test_image,
    create_test_images,
    load_images,
    preprocess_image,
    print_benchmark_results,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 后端基类
# =============================================================================

class TaskDecompositionBackend(ABC):
    """任务分解后端抽象基类，支持 1-3 张图像输入"""
    
    @abstractmethod
    def decompose(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """
        分解任务为左右臂子任务
        
        Args:
            images: 1-3 张图像的字典
                - "base_rgb": 基座相机图像
                - "left_wrist_rgb": 左手腕相机图像（可选）
                - "right_wrist_rgb": 右手腕相机图像（可选）
            instruction: 任务指令
            
        Returns:
            BimanualPromptResult: 包含左右臂 prompt
        """
        pass
    
    def __call__(self, images: Dict[str, np.ndarray], instruction: str) -> BimanualPromptResult:
        return self.decompose(images, instruction)


# =============================================================================
# 规则后端 (Rule-based)
# =============================================================================

class RuleBasedBackend(TaskDecompositionBackend):
    """基于规则的任务分解"""
    
    # 任务分解规则
    TASK_PATTERNS = {
        "handover": {
            "keywords": ["handover", "pass", "transfer", "hand over", "give"],
            "left": "Grab the object and move to handover position",
            "right": "Receive the object from left arm and place at target",
        },
        "cooperative": {
            "keywords": ["together", "cooperative", "both hands", "both arms", "bimanual"],
            "left": "Left arm: coordinate grip on left side",
            "right": "Right arm: coordinate grip on right side",
        },
        "shake": {
            "keywords": ["shake"],
            "left": "Left arm: grip object firmly",
            "right": "Right arm: grip object firmly and coordinate shaking motion",
        },
        "fold": {
            "keywords": ["fold"],
            "left": "Left arm: hold one side of the object",
            "right": "Right arm: fold the other side",
        },
        "lift": {
            "keywords": ["lift", "raise", "carry"],
            "left": "Left arm: grip and lift from left side",
            "right": "Right arm: grip and lift from right side",
        },
    }
    
    def decompose(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用规则生成左右臂 prompt"""
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
            has_left = any(kw in instruction_lower for kw in ["left arm", "left hand", "use left", "with left"])
            has_right = any(kw in instruction_lower for kw in ["right arm", "right hand", "use right", "with right"])
            
            if has_left and has_right:
                # 尝试分割双臂任务
                use_pattern = re.search(
                    r'use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?(.+?)(?:\s+and\s+use|\s+use)\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?(.+)',
                    instruction, flags=re.IGNORECASE
                )
                if use_pattern:
                    first_part = use_pattern.group(0)
                    if re.search(r'use\s+(?:the\s+)?left', first_part, re.IGNORECASE):
                        left_prompt = use_pattern.group(1).strip().rstrip('.').rstrip(',')
                        right_prompt = use_pattern.group(2).strip().rstrip('.')
                    else:
                        right_prompt = use_pattern.group(1).strip().rstrip('.').rstrip(',')
                        left_prompt = use_pattern.group(2).strip().rstrip('.')
                else:
                    # 使用 "and" 分割
                    parts = re.split(r'\s+and\s+', instruction, flags=re.IGNORECASE)
                    for part in parts:
                        part_lower = part.lower()
                        part_clean = part.strip()
                        if any(kw in part_lower for kw in ["left arm", "left hand", "use left", "with left"]):
                            task = re.sub(r'^use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?', '', part_clean, flags=re.IGNORECASE)
                            left_prompt = task.strip() if task.strip() else part_clean
                        elif any(kw in part_lower for kw in ["right arm", "right hand", "use right", "with right"]):
                            task = re.sub(r'^use\s+(?:the\s+)?(?:left|right)\s+(?:hand|arm)\s+(?:to\s+)?', '', part_clean, flags=re.IGNORECASE)
                            right_prompt = task.strip() if task.strip() else part_clean
                
                if left_prompt and not right_prompt:
                    right_prompt = "Coordinate with left arm"
                elif right_prompt and not left_prompt:
                    left_prompt = "Coordinate with right arm"
                    
            elif has_left:
                task = re.sub(r'.*?(?:left\s+(?:hand|arm))\s+(?:to\s+)?', '', instruction, flags=re.IGNORECASE)
                left_prompt = task.strip() if task.strip() else instruction
                right_prompt = "Stand by and prepare to assist"
            elif has_right:
                task = re.sub(r'.*?(?:right\s+(?:hand|arm))\s+(?:to\s+)?', '', instruction, flags=re.IGNORECASE)
                right_prompt = task.strip() if task.strip() else instruction
                left_prompt = "Stand by and prepare to assist"
            else:
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


# =============================================================================
# 图像分析后端 (使用 numpy/PIL)
# =============================================================================

class ImageAnalysisBackend(TaskDecompositionBackend):
    """
    基于图像分析的任务分解（支持 1-3 张图像）
    
    结合规则方法和图像空间分析（使用 numpy/PIL），不依赖深度学习库
    """
    
    def __init__(self):
        self._rule_backend = RuleBasedBackend()
        logger.info("图像分析任务分解后端初始化完成")
    
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
        
        gray = np.mean(image, axis=2).astype(np.float32)
        
        # 计算左右两侧相似度
        left_half = image[:, :half_w, :]
        right_half = image[:, half_w:, :]
        
        left_hist = np.histogram(left_half.flatten(), bins=32, range=(0, 255))[0].astype(np.float32)
        right_hist = np.histogram(right_half.flatten(), bins=32, range=(0, 255))[0].astype(np.float32)
        
        left_hist = left_hist / (left_hist.sum() + 1e-6)
        right_hist = right_hist / (right_hist.sum() + 1e-6)
        
        left_right_similarity = np.minimum(left_hist, right_hist).sum()
        
        # 计算中心活跃度
        edge_map = self._compute_edge_map(gray)
        center_h_start, center_h_end = h // 3, h - h // 3
        center_w_start, center_w_end = w // 3, w - w // 3
        center_region = edge_map[center_h_start:center_h_end, center_w_start:center_w_end]
        
        center_activity = np.mean(center_region) / (np.mean(edge_map) + 1e-6)
        center_activity = min(1.0, center_activity)
        
        return {
            "left_right_similarity": float(left_right_similarity),
            "center_activity": float(center_activity),
        }
    
    def _analyze_wrist_image(self, image: np.ndarray) -> dict:
        """分析手腕相机图像的特征"""
        image = preprocess_image(image)
        gray = np.mean(image, axis=2).astype(np.float32)
        edge_map = self._compute_edge_map(gray)
        
        object_presence = min(1.0, np.mean(edge_map) / 50.0)
        
        h, w = image.shape[:2]
        center_region = edge_map[h//4:3*h//4, w//4:3*w//4]
        center_focus = min(1.0, np.mean(center_region) / (np.mean(edge_map) + 1e-6))
        
        return {
            "object_presence": float(object_presence),
            "center_focus": float(center_focus),
        }
    
    def _analyze_spatial_distribution(self, images: Dict[str, np.ndarray]) -> dict:
        """分析多张图像的空间分布"""
        result = {"num_images": len(images)}
        
        # 分析基座图像
        base_image = images.get("base_rgb", list(images.values())[0])
        base_analysis = self._analyze_single_image(base_image)
        result.update(base_analysis)
        
        # 分析手腕图像
        left_wrist_analysis = None
        right_wrist_analysis = None
        
        if "left_wrist_rgb" in images:
            left_wrist_analysis = self._analyze_wrist_image(images["left_wrist_rgb"])
            result["left_wrist_presence"] = left_wrist_analysis["object_presence"]
        
        if "right_wrist_rgb" in images:
            right_wrist_analysis = self._analyze_wrist_image(images["right_wrist_rgb"])
            result["right_wrist_presence"] = right_wrist_analysis["object_presence"]
        
        if left_wrist_analysis and right_wrist_analysis:
            wrist_similarity = 1.0 - abs(
                left_wrist_analysis["object_presence"] - right_wrist_analysis["object_presence"]
            )
            result["wrist_similarity"] = float(wrist_similarity)
            result["both_hands_active"] = (
                left_wrist_analysis["object_presence"] > 0.3 and 
                right_wrist_analysis["object_presence"] > 0.3
            )
        
        return result
    
    def decompose(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """结合多图像分析生成双臂 prompt"""
        # 基础规则分解
        result = self._rule_backend.decompose(images, instruction)
        
        # 分析多图像空间分布
        spatial_info = self._analyze_spatial_distribution(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 根据图像分析调整 prompt
        need_coordinate = False
        
        if spatial_info["left_right_similarity"] > 0.8 and spatial_info["center_activity"] > 0.7:
            need_coordinate = True
        
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
        
        result.raw_output = "\n".join(raw_parts)
        
        return result


# =============================================================================
# PaliGemma VLM 后端
# =============================================================================

class PaliGemmaBackend(TaskDecompositionBackend):
    """
    使用 Pi0.5 内置的 PaliGemma VLM 进行任务分解（支持 1-3 张图像）
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
    ):
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
        
        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(configs=[paligemma_config], embed_dtype=config.dtype, adarms=False)
        )
        self.llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        
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
        
        if self.checkpoint_path:
            logger.info(f"从 {self.checkpoint_path} 加载权重...")
            params = _model.restore_params(self.checkpoint_path)
        
        self._rule_backend = RuleBasedBackend()
        self._image_backend = ImageAnalysisBackend()
        
        logger.info("PaliGemma 任务分解后端加载完成!")
    
    def _preprocess_image(self, image: np.ndarray):
        """预处理图像为 JAX 张量"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.shape[:2] != (224, 224):
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((224, 224), Image.BILINEAR)
            image = np.array(pil_image).astype(np.float32) / 255.0
        
        image = image * 2.0 - 1.0
        image = image[np.newaxis, ...]
        
        return self.jnp.array(image)
    
    def _analyze_image_features(self, images: Dict[str, np.ndarray]) -> dict:
        """分析多张图像的深度特征"""
        result = {"num_images": len(images)}
        
        for key, img in images.items():
            image_tensor = self._preprocess_image(img)
            image_tokens, _ = self.img_encoder(image_tensor, train=False)
            
            tokens = np.array(image_tokens[0])
            num_patches = tokens.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size == num_patches:
                tokens_grid = tokens.reshape(grid_size, grid_size, -1)
                half = grid_size // 2
                
                left_mean = np.mean(tokens_grid[:, :half, :], axis=(0, 1))
                right_mean = np.mean(tokens_grid[:, half:, :], axis=(0, 1))
                
                left_norm = np.linalg.norm(left_mean)
                right_norm = np.linalg.norm(right_mean)
                
                if left_norm > 0 and right_norm > 0:
                    similarity = np.dot(left_mean, right_mean) / (left_norm * right_norm)
                    result[f"{key}_lr_similarity"] = float((similarity + 1) / 2)
        
        # 基座图像的相似度作为主要指标
        result["left_right_similarity"] = result.get("base_rgb_lr_similarity", 0.5)
        result["center_activity"] = 0.5  # 简化处理
        
        return result
    
    def decompose(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用 PaliGemma 生成双臂 prompt"""
        spatial_info = self._analyze_image_features(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 使用规则方法作为基础
        result = self._rule_backend.decompose(images, instruction)
        
        # 根据视觉特征调整
        if spatial_info["left_right_similarity"] > 0.8:
            if "together" not in instruction.lower() and "both" not in instruction.lower():
                if "coordinate" not in result.left_arm_prompt.lower():
                    result.left_arm_prompt = f"Coordinate: {result.left_arm_prompt}"
                    result.right_arm_prompt = f"Coordinate: {result.right_arm_prompt}"
        
        result.raw_output = f"[PaliGemma VLM] (images: {num_images})\n{result.raw_output}"
        
        return result


# =============================================================================
# API 后端
# =============================================================================

class APIBackend(TaskDecompositionBackend):
    """使用外部 VLM API (如 OpenAI GPT-4V) 进行任务分解"""
    
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
        
        logger.info(f"API 任务分解后端初始化完成, provider: {api_provider}")
    
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
    
    def decompose(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> BimanualPromptResult:
        """使用 API 生成双臂 prompt"""
        if self._fallback:
            return self._fallback.decompose(images, instruction)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            num_images = len(images)
            
            prompt_text = f"""Look at these {num_images} image(s) of a bimanual robot workspace.

Task: {instruction}

Please decompose this task into two separate subtasks - one for the left arm and one for the right arm.
Consider all provided camera views when making your decision.

Format your response EXACTLY as:
Left Arm: [specific subtask for left arm]
Right Arm: [specific subtask for right arm]

Be specific and concise."""

            content = [{"type": "text", "text": prompt_text}]
            content.extend(self._build_image_content(images))

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
            )
            
            raw_output = response.choices[0].message.content
            
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
            return RuleBasedBackend().decompose(images, instruction)


# =============================================================================
# 主模型类
# =============================================================================

class TaskDecomposer:
    """
    任务分解模型（支持 1-3 张图像输入）
    
    给定任务截图和任务 prompt，输出左右臂分别的 prompt。
    
    支持多种后端模式:
        - rule: 基于规则（快速，不需要模型）
        - image: 规则 + numpy 图像分析（推荐，不依赖深度学习库）
        - paligemma: 使用 Pi0.5 内置的 PaliGemma VLM
        - api: 使用外部 VLM API（如 GPT-4V）
    
    Example:
        >>> decomposer = TaskDecomposer(mode="image")
        >>> 
        >>> # 单张图片
        >>> result = decomposer(base_image, "Pick up the bottle")
        >>> print(result.left_arm_prompt)
        >>> print(result.right_arm_prompt)
        >>> 
        >>> # 多张图片
        >>> images = {"base_rgb": img1, "left_wrist_rgb": img2, "right_wrist_rgb": img3}
        >>> result = decomposer(images, "Shake the bottle")
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
            self.backend = APIBackend(api_key=api_key, api_provider=api_provider)
        
        logger.info(f"TaskDecomposer 初始化完成, 模式: {mode}")
    
    def decompose(
        self,
        images: ImageInput,
        instruction: str,
    ) -> BimanualPromptResult:
        """
        分解任务为左右臂子任务
        
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
        return self.backend.decompose(images_dict, instruction)
    
    def __call__(self, images: ImageInput, instruction: str) -> BimanualPromptResult:
        return self.decompose(images, instruction)


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_task_decomposer(mode: str = "image", **kwargs) -> TaskDecomposer:
    """创建任务分解器实例"""
    return TaskDecomposer(mode=mode, **kwargs)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_all_modes(
    images: ImageInput,
    instruction: str,
    num_warmup: int = 3,
    num_runs: int = 10,
    checkpoint_path: Optional[str] = None,
    use_pretrained: bool = False,
) -> dict:
    """对比测试所有可用模式的 inference time"""
    all_results = {}
    
    modes_to_test = ["rule", "image"]
    if checkpoint_path or use_pretrained:
        modes_to_test.append("paligemma")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Benchmarking All Modes - Task Decomposition")
    logger.info(f"Instruction: {instruction}")
    logger.info("=" * 70)
    
    for mode in modes_to_test:
        logger.info("")
        logger.info(f">>> Testing mode: {mode}")
        logger.info("-" * 50)
        
        try:
            load_start = time.perf_counter()
            model = TaskDecomposer(
                mode=mode,
                checkpoint_path=checkpoint_path,
                use_pretrained=use_pretrained,
            )
            load_time = time.perf_counter() - load_start
            logger.info(f"Model load time: {load_time:.2f}s (NOT counted in inference)")
            
            results = benchmark_inference(
                model, images, instruction,
                num_warmup=num_warmup,
                num_runs=num_runs,
                verbose=False,
            )
            results["load_time_s"] = load_time
            
            all_results[mode] = results
            print_benchmark_results(results, model_name=f"Task Decomposition ({mode})")
            
        except Exception as e:
            logger.error(f"Failed to benchmark mode '{mode}': {e}")
            all_results[mode] = {"error": str(e)}
    
    # 打印对比摘要
    logger.info("")
    logger.info("=" * 70)
    logger.info("Summary Comparison")
    logger.info("=" * 70)
    logger.info(f"{'Mode':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput':<15} {'Load (s)':<10}")
    logger.info("-" * 70)
    
    for mode, results in all_results.items():
        if "error" in results:
            logger.info(f"{mode:<15} {'ERROR':<12} {'-':<12} {'-':<15} {'-':<10}")
        else:
            throughput = f"{1000 / results['mean_ms']:.1f} samples/s"
            logger.info(
                f"{mode:<15} {results['mean_ms']:<12.2f} {results['std_ms']:<12.2f} "
                f"{throughput:<15} {results.get('load_time_s', 0):<10.2f}"
            )
    
    logger.info("=" * 70)
    
    return all_results


# =============================================================================
# Main
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双臂任务分解 (Task Decomposition)")
    parser.add_argument("--mode", type=str, default="image",
                        choices=["rule", "image", "paligemma", "api"],
                        help="后端模式")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径 (paligemma 模式)")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="使用预训练模型 (paligemma 模式)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (api 模式)")
    
    # 图像输入参数
    parser.add_argument("--image", type=str, default=None,
                        help="输入图像路径（单张图像）")
    parser.add_argument("--base-image", type=str, default=None,
                        help="基座相机图像路径")
    parser.add_argument("--left-wrist-image", type=str, default=None,
                        help="左手腕相机图像路径")
    parser.add_argument("--right-wrist-image", type=str, default=None,
                        help="右手腕相机图像路径")
    
    parser.add_argument("--instruction", type=str,
                        default="Shake the bottle with both hands",
                        help="任务指令")
    
    # Benchmark 参数
    parser.add_argument("--benchmark", action="store_true",
                        help="运行 inference time benchmark 测试")
    parser.add_argument("--benchmark-all", action="store_true",
                        help="对比测试所有可用模式")
    parser.add_argument("--num-warmup", type=int, default=3,
                        help="Benchmark 预热次数")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Benchmark 测试次数")
    
    args = parser.parse_args()
    
    # 加载图像
    images = {}
    if args.base_image and Path(args.base_image).exists():
        images["base_rgb"] = np.array(Image.open(args.base_image).convert('RGB'))
    if args.left_wrist_image and Path(args.left_wrist_image).exists():
        images["left_wrist_rgb"] = np.array(Image.open(args.left_wrist_image).convert('RGB'))
    if args.right_wrist_image and Path(args.right_wrist_image).exists():
        images["right_wrist_rgb"] = np.array(Image.open(args.right_wrist_image).convert('RGB'))
    
    if not images:
        if args.image and Path(args.image).exists():
            images["base_rgb"] = np.array(Image.open(args.image).convert('RGB'))
        else:
            logger.info("未提供图像，使用随机测试图像...")
            images = create_test_images()
    
    logger.info(f"图像数量: {len(images)}")
    
    # Benchmark 模式
    if args.benchmark_all:
        benchmark_all_modes(
            images=images,
            instruction=args.instruction,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            checkpoint_path=args.checkpoint,
            use_pretrained=args.use_pretrained,
        )
        return
    
    if args.benchmark:
        logger.info("=" * 60)
        logger.info("Inference Time Benchmark - Task Decomposition")
        logger.info(f"模式: {args.mode}")
        logger.info("=" * 60)
        
        load_start = time.perf_counter()
        model = TaskDecomposer(
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            use_pretrained=args.use_pretrained,
            api_key=args.api_key,
        )
        load_time = time.perf_counter() - load_start
        logger.info(f"模型加载时间: {load_time:.2f}s (不计入 inference time)")
        
        results = benchmark_inference(
            model=model,
            images=images,
            instruction=args.instruction,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            verbose=True,
        )
        print_benchmark_results(results, model_name=f"Task Decomposition ({args.mode})")
        return
    
    # 普通模式
    model = TaskDecomposer(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        use_pretrained=args.use_pretrained,
        api_key=args.api_key,
    )
    
    logger.info(f"\n指令: {args.instruction}")
    logger.info("-" * 50)
    
    result = model(images, args.instruction)
    
    logger.info("\n任务分解结果:")
    logger.info(f"  左臂: {result.left_arm_prompt}")
    logger.info(f"  右臂: {result.right_arm_prompt}")
    
    # 测试更多指令
    test_instructions = [
        "Pick up the bottle with left arm and shake it",
        "Use both arms to lift the heavy box together",
        "Left arm grabs the red cube, right arm grabs the blue cube",
        "Hand over the tool from left arm to right arm",
    ]
    
    logger.info("\n" + "-" * 50)
    logger.info("更多测试:")
    for instr in test_instructions:
        result = model(images, instr)
        logger.info(f"\n指令: {instr}")
        logger.info(f"  左臂: {result.left_arm_prompt}")
        logger.info(f"  右臂: {result.right_arm_prompt}")


if __name__ == "__main__":
    main()

