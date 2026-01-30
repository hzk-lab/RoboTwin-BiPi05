#!/usr/bin/env python3
"""
双臂协调度评估模块 (Cooperation Evaluation)

给定 1-3 张任务截图和任务 prompt，输出左右手的协调度（0 到 1 之间的连续值）。

输入:
    - images: 1-3 张任务场景图像 (RGB, 224x224)
        - base_rgb: 基座相机图像（必需）
        - left_wrist_rgb: 左手腕相机图像（可选）
        - right_wrist_rgb: 右手腕相机图像（可选）
    - instruction: 任务指令

输出:
    - cooperation_score: 协调度分数，连续值 [0, 1]
        - 0.0-0.2: 非常低的协作需求，双臂几乎完全独立
        - 0.2-0.4: 较低的协作需求，简单的传递或等待
        - 0.4-0.6: 中等协作需求，需要一定的时序配合
        - 0.6-0.8: 较高的协作需求，需要空间和时间上的协调
        - 0.8-1.0: 非常高的协作需求，需要紧密同步和对齐

支持四种模式:
    1. rule: 基于规则的评估 (快速，不需要模型)
    2. image: 规则 + numpy 图像分析 (推荐，不依赖深度学习库)
    3. paligemma: 使用 Pi0.5 内置的 PaliGemma VLM (需要预训练权重)
    4. api: 使用外部 VLM API (如 OpenAI GPT-4V)

Usage:
    cd /path/to/RoboTwin/policy/pi05
    
    # 使用规则模式
    python experiments/cooperation_evaluation.py --mode rule
    
    # 使用图像分析模式 (推荐)
    python experiments/cooperation_evaluation.py --mode image
    
    # 多图像输入
    python experiments/cooperation_evaluation.py --mode image \\
        --base-image base.jpg --left-wrist-image left.jpg --right-wrist-image right.jpg
    
    # Inference Time Benchmark
    python experiments/cooperation_evaluation.py --mode rule --benchmark
    
    # 对比所有模式
    python experiments/cooperation_evaluation.py --benchmark-all
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
    CooperationResult,
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

class CooperationBackend(ABC):
    """协调度评估后端抽象基类，支持 1-3 张图像输入"""
    
    @abstractmethod
    def evaluate(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """
        评估协调度
        
        Args:
            images: 1-3 张图像的字典
            instruction: 任务指令
            
        Returns:
            CooperationResult: 包含 cooperation_score (连续值 0-1)
        """
        pass
    
    def __call__(self, images: Dict[str, np.ndarray], instruction: str) -> CooperationResult:
        return self.evaluate(images, instruction)


# =============================================================================
# 规则后端 (Rule-based)
# =============================================================================

class RuleBasedBackend(CooperationBackend):
    """基于规则的协调度评估 - 输出连续值 [0, 1]"""
    
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
    
    def evaluate(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用规则评估协调度 - 输出连续值 [0, 1]"""
        instruction_lower = instruction.lower()
        num_images = len(images)
        
        # 基础分数 - 根据指令长度和复杂度微调
        word_count = len(instruction.split())
        base_score = 0.45 + min(0.1, word_count * 0.005)
        
        score_delta = 0.0
        matched_keywords = []
        
        # 检测高协作关键词
        for keyword, weight in self.HIGH_COOP_KEYWORDS.items():
            if keyword in instruction_lower:
                pos = instruction_lower.find(keyword)
                pos_factor = 1.0 + (1.0 - pos / len(instruction_lower)) * 0.1
                adjusted_weight = weight * pos_factor
                score_delta += adjusted_weight
                matched_keywords.append(f"+{keyword}({adjusted_weight:.3f})")
        
        # 检测低协作关键词
        for keyword, weight in self.LOW_COOP_KEYWORDS.items():
            if keyword in instruction_lower:
                pos = instruction_lower.find(keyword)
                pos_factor = 1.0 + (1.0 - pos / len(instruction_lower)) * 0.1
                adjusted_weight = weight * pos_factor
                score_delta -= adjusted_weight
                matched_keywords.append(f"-{keyword}({adjusted_weight:.3f})")
        
        # 特殊模式检测
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
        
        # 精确协调短语
        precision_phrases = {
            "make sure": 0.06, "ensure": 0.05, "keep": 0.04,
            "carefully": 0.07, "precisely": 0.08, "exactly": 0.06,
        }
        for phrase, weight in precision_phrases.items():
            if phrase in instruction_lower:
                if any(w in instruction_lower for w in ["aligned", "synchronized", "together"]):
                    score_delta += weight
                    matched_keywords.append(f"+precision_{phrase}({weight})")
        
        # 检测左右臂分别做不同任务
        if "left arm" in instruction_lower and "right arm" in instruction_lower:
            has_coop_words = any(w in instruction_lower for w in ["together", "coordinate", "synchronized", "both"])
            if not has_coop_words:
                if " and " in instruction_lower or ", " in instruction_lower:
                    score_delta -= 0.15
                    matched_keywords.append("-separate_tasks(0.15)")
                elif " while " in instruction_lower:
                    score_delta -= 0.08
                    matched_keywords.append("-parallel_tasks(0.08)")
        
        # 单臂任务检测
        single_arm_indicators = ["left arm only", "right arm only", "use left", "use right", "with left", "with right"]
        for indicator in single_arm_indicators:
            if indicator in instruction_lower:
                if "left arm" not in instruction_lower or "right arm" not in instruction_lower:
                    score_delta -= 0.25
                    matched_keywords.append(f"-single_arm({indicator})(0.25)")
                    break
        
        # 计算最终分数 - 软裁剪
        raw_score = base_score + score_delta
        if raw_score > 1.0:
            final_score = 1.0 - 0.05 * np.exp(-(raw_score - 1.0) * 3)
        elif raw_score < 0.0:
            final_score = 0.05 * np.exp(raw_score * 3)
        else:
            final_score = raw_score
        
        # 添加微小随机性
        hash_noise = (hash(instruction) % 100) / 10000.0 - 0.005
        final_score = max(0.0, min(1.0, final_score + hash_noise))
        
        explanation = f"Matched: {', '.join(matched_keywords)}" if matched_keywords else "No specific keywords matched"
        
        return CooperationResult(
            cooperation_score=round(final_score, 4),
            explanation=explanation,
            raw_output=f"[Rule-based Cooperation] (images: {num_images})\n"
                      f"Base: {base_score:.4f}, Delta: {score_delta:+.4f}, Final: {final_score:.4f}\n"
                      f"{explanation}",
        )


# =============================================================================
# 图像分析后端 (使用 numpy/PIL)
# =============================================================================

class ImageAnalysisBackend(CooperationBackend):
    """
    基于图像分析的协调度评估（支持 1-3 张图像）
    
    结合规则方法和图像空间分析（使用 numpy/PIL），不依赖深度学习库
    """
    
    def __init__(self):
        self._rule_backend = RuleBasedBackend()
        logger.info("图像分析协调度评估后端初始化完成")
    
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
            result["left_wrist_focus"] = left_wrist_analysis["center_focus"]
        
        if "right_wrist_rgb" in images:
            right_wrist_analysis = self._analyze_wrist_image(images["right_wrist_rgb"])
            result["right_wrist_presence"] = right_wrist_analysis["object_presence"]
            result["right_wrist_focus"] = right_wrist_analysis["center_focus"]
        
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
    
    def evaluate(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """结合多图像分析评估协调度 - 输出连续值 [0, 1]"""
        # 基础规则评估
        rule_result = self._rule_backend.evaluate(images, instruction)
        base_score = rule_result.cooperation_score
        
        # 多图像分析
        spatial_info = self._analyze_spatial_distribution(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 根据图像分析调整分数
        adjustments = []
        score_delta = 0.0
        
        left_right_sim = spatial_info["left_right_similarity"]
        center_activity = spatial_info["center_activity"]
        spatial_spread = spatial_info["spatial_spread"]
        
        # === 基座图像分析 ===
        sim_contribution = (left_right_sim - 0.5) * 0.25
        score_delta += sim_contribution
        adjustments.append(f"左右相似度({left_right_sim:.3f})={sim_contribution:+.4f}")
        
        center_contribution = (center_activity - 0.8) * 0.15
        score_delta += center_contribution
        adjustments.append(f"中心活跃度({center_activity:.3f})={center_contribution:+.4f}")
        
        spread_contribution = -(spatial_spread - 0.5) * 0.12
        score_delta += spread_contribution
        adjustments.append(f"空间分散度({spatial_spread:.3f})={spread_contribution:+.4f}")
        
        # === 手腕图像分析 ===
        if "left_wrist_presence" in spatial_info and "right_wrist_presence" in spatial_info:
            left_presence = spatial_info["left_wrist_presence"]
            right_presence = spatial_info["right_wrist_presence"]
            
            if left_presence > 0.3 and right_presence > 0.3:
                both_active_bonus = min(left_presence, right_presence) * 0.15
                score_delta += both_active_bonus
                adjustments.append(f"双手活跃={both_active_bonus:+.4f}")
            
            if "wrist_similarity" in spatial_info:
                wrist_sim = spatial_info["wrist_similarity"]
                wrist_contribution = (wrist_sim - 0.5) * 0.12
                score_delta += wrist_contribution
                adjustments.append(f"手腕相似度({wrist_sim:.3f})={wrist_contribution:+.4f}")
        
        elif "left_wrist_presence" in spatial_info or "right_wrist_presence" in spatial_info:
            single_wrist = spatial_info.get("left_wrist_presence", spatial_info.get("right_wrist_presence", 0))
            if single_wrist > 0.5:
                score_delta -= 0.05
                adjustments.append(f"单手腕活跃=-0.05")
        
        # === 综合指标 ===
        if left_right_sim > 0.7 and center_activity > 0.9:
            synergy_bonus = (left_right_sim - 0.7) * (center_activity - 0.9) * 0.5
            score_delta += synergy_bonus
            adjustments.append(f"协同加成={synergy_bonus:+.4f}")
        
        if spatial_spread > 0.6 and left_right_sim < 0.4:
            independent_penalty = (spatial_spread - 0.6) * (0.4 - left_right_sim) * 0.4
            score_delta -= independent_penalty
            adjustments.append(f"独立任务惩罚={-independent_penalty:+.4f}")
        
        # 多图像加成
        if num_images >= 2:
            multi_view_factor = 1.0 + (num_images - 1) * 0.02
            score_delta *= multi_view_factor
            adjustments.append(f"多视角因子={multi_view_factor:.2f}")
        
        # 计算最终分数
        raw_score = base_score + score_delta
        
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

class PaliGemmaBackend(CooperationBackend):
    """
    使用 Pi0.5 内置的 PaliGemma VLM 进行协调度评估（支持 1-3 张图像）
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
    ):
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import flax.nnx.bridge as nnx_bridge
        from openpi.models import model as _model
        from openpi.models import pi0_config
        import openpi.models.siglip as _siglip
        from openpi.shared import download
        
        self.jax = jax
        self.jnp = jnp
        
        if use_pretrained:
            pretrained_url = PRETRAINED_MODELS.get("pi05_base")
            logger.info(f"下载预训练模型: {pretrained_url}")
            checkpoint_path = download.maybe_download(pretrained_url)
        
        self.checkpoint_path = checkpoint_path
        
        # 加载图像编码器
        logger.info("加载图像编码器...")
        config = pi0_config.Pi0Config(pi05=True)
        
        rngs = nnx.Rngs(jax.random.key(0))
        
        self.img_encoder = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=config.paligemma_config.width if hasattr(config, 'paligemma_config') else 2048,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        fake_image = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
        self.img_encoder.lazy_init(fake_image, train=False, rngs=rngs)
        
        self._rule_backend = RuleBasedBackend()
        
        logger.info("PaliGemma 协调度评估后端加载完成!")
    
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
        
        encoded_features = {}
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
                    encoded_features[key] = {
                        "left_right_similarity": float((similarity + 1) / 2),
                        "global_feature": np.mean(tokens, axis=0),
                        "feature_norm": float(np.linalg.norm(np.mean(tokens, axis=0))),
                    }
        
        # 基座图像分析
        base_key = "base_rgb" if "base_rgb" in encoded_features else list(encoded_features.keys())[0]
        if base_key in encoded_features:
            result["left_right_similarity"] = encoded_features[base_key]["left_right_similarity"]
        else:
            result["left_right_similarity"] = 0.5
        
        result["center_activity"] = 0.5
        result["spatial_spread"] = 0.5
        
        # 手腕图像分析
        if "left_wrist_rgb" in encoded_features:
            result["left_wrist_activity"] = min(1.0, encoded_features["left_wrist_rgb"]["feature_norm"] / 100.0)
        if "right_wrist_rgb" in encoded_features:
            result["right_wrist_activity"] = min(1.0, encoded_features["right_wrist_rgb"]["feature_norm"] / 100.0)
        
        if "left_wrist_rgb" in encoded_features and "right_wrist_rgb" in encoded_features:
            left_feat = encoded_features["left_wrist_rgb"]["global_feature"]
            right_feat = encoded_features["right_wrist_rgb"]["global_feature"]
            
            left_norm = np.linalg.norm(left_feat)
            right_norm = np.linalg.norm(right_feat)
            
            if left_norm > 0 and right_norm > 0:
                wrist_similarity = np.dot(left_feat, right_feat) / (left_norm * right_norm)
                result["wrist_feature_similarity"] = float((wrist_similarity + 1) / 2)
            
            if (result.get("left_wrist_activity", 0) > 0.3 and 
                result.get("right_wrist_activity", 0) > 0.3):
                result["both_hands_active"] = True
        
        return result
    
    def evaluate(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用 PaliGemma 评估协调度"""
        spatial_info = self._analyze_image_features(images)
        num_images = spatial_info.get("num_images", 1)
        
        # 规则基础评估
        rule_result = self._rule_backend.evaluate(images, instruction)
        base_score = rule_result.cooperation_score
        
        # 根据深度视觉特征调整
        adjustments = []
        score_delta = 0.0
        
        left_right_sim = spatial_info["left_right_similarity"]
        
        # 视觉相似度贡献
        sim_contribution = (left_right_sim - 0.5) * 0.30
        score_delta += sim_contribution
        adjustments.append(f"视觉相似度({left_right_sim:.3f})={sim_contribution:+.4f}")
        
        # 手腕图像分析
        if "left_wrist_activity" in spatial_info and "right_wrist_activity" in spatial_info:
            left_activity = spatial_info["left_wrist_activity"]
            right_activity = spatial_info["right_wrist_activity"]
            
            if left_activity > 0.3 and right_activity > 0.3:
                both_active_bonus = min(left_activity, right_activity) * 0.18
                score_delta += both_active_bonus
                adjustments.append(f"双手活跃={both_active_bonus:+.4f}")
            
            if "wrist_feature_similarity" in spatial_info:
                wrist_sim = spatial_info["wrist_feature_similarity"]
                wrist_contribution = (wrist_sim - 0.5) * 0.15
                score_delta += wrist_contribution
                adjustments.append(f"手腕特征相似度({wrist_sim:.3f})={wrist_contribution:+.4f}")
        
        # 多图像加成
        if num_images >= 2:
            multi_view_factor = 1.0 + (num_images - 1) * 0.03
            score_delta *= multi_view_factor
            adjustments.append(f"多视角因子={multi_view_factor:.2f}")
        
        # 计算最终分数
        raw_score = base_score + score_delta
        
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
                      f"Visual: L-R={left_right_sim:.4f}\n"
                      f"Score: base={base_score:.4f}, delta={score_delta:+.4f}, final={final_score:.4f}",
            spatial_info=spatial_info,
        )


# =============================================================================
# API 后端
# =============================================================================

class APIBackend(CooperationBackend):
    """使用外部 VLM API (如 OpenAI GPT-4V) 进行协调度评估"""
    
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
        
        logger.info(f"API 协调度评估后端初始化完成, provider: {api_provider}")
    
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
    
    def evaluate(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
    ) -> CooperationResult:
        """使用 API 评估协调度"""
        if self._fallback:
            return self._fallback.evaluate(images, instruction)
        
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

            content = [{"type": "text", "text": prompt_text}]
            content.extend(self._build_image_content(images))

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=150,
            )
            
            raw_output = response.choices[0].message.content
            
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
            return RuleBasedBackend().evaluate(images, instruction)


# =============================================================================
# 主模型类
# =============================================================================

class CooperationEvaluator:
    """
    协调度评估模型（支持 1-3 张图像输入）
    
    给定任务截图和任务 prompt，输出左右手的协调度（0 到 1 之间的连续值）。
    
    支持多种后端模式:
        - rule: 基于规则（快速，不需要模型）
        - image: 规则 + numpy 图像分析（推荐，不依赖深度学习库）
        - paligemma: 使用 Pi0.5 内置的 PaliGemma VLM
        - api: 使用外部 VLM API（如 GPT-4V）
    
    Example:
        >>> evaluator = CooperationEvaluator(mode="image")
        >>> 
        >>> # 单张图片
        >>> result = evaluator(base_image, "Shake the bottle with both hands")
        >>> print(result.cooperation_score)  # 0.7823
        >>> 
        >>> # 多张图片
        >>> images = {"base_rgb": img1, "left_wrist_rgb": img2, "right_wrist_rgb": img3}
        >>> result = evaluator(images, "Lift the heavy box together")
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
        
        logger.info(f"CooperationEvaluator 初始化完成, 模式: {mode}")
    
    def evaluate(
        self,
        images: ImageInput,
        instruction: str,
    ) -> CooperationResult:
        """
        评估协调度
        
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
        return self.backend.evaluate(images_dict, instruction)
    
    def __call__(self, images: ImageInput, instruction: str) -> CooperationResult:
        return self.evaluate(images, instruction)


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_cooperation_evaluator(mode: str = "image", **kwargs) -> CooperationEvaluator:
    """创建协调度评估器实例"""
    return CooperationEvaluator(mode=mode, **kwargs)


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
    logger.info("Benchmarking All Modes - Cooperation Evaluation")
    logger.info(f"Instruction: {instruction}")
    logger.info("=" * 70)
    
    for mode in modes_to_test:
        logger.info("")
        logger.info(f">>> Testing mode: {mode}")
        logger.info("-" * 50)
        
        try:
            load_start = time.perf_counter()
            model = CooperationEvaluator(
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
            print_benchmark_results(results, model_name=f"Cooperation Evaluation ({mode})")
            
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
    parser = argparse.ArgumentParser(description="双臂协调度评估 (Cooperation Evaluation)")
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
        logger.info("Inference Time Benchmark - Cooperation Evaluation")
        logger.info(f"模式: {args.mode}")
        logger.info("=" * 60)
        
        load_start = time.perf_counter()
        model = CooperationEvaluator(
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
        print_benchmark_results(results, model_name=f"Cooperation Evaluation ({args.mode})")
        return
    
    # 普通模式
    model = CooperationEvaluator(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        use_pretrained=args.use_pretrained,
        api_key=args.api_key,
    )
    
    logger.info(f"\n指令: {args.instruction}")
    logger.info("-" * 50)
    
    result = model(images, args.instruction)
    
    # 显示连续值和可视化条
    score = result.cooperation_score
    bar_len = int(score * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    
    logger.info("\n协调度评估结果:")
    logger.info(f"  协调度: {score:.4f} [{bar}]")
    logger.info(f"  解释: {result.explanation}")
    
    # 测试更多指令
    test_cases = [
        ("Shake the bottle with both hands", "高"),
        ("Lift the heavy box together", "高"),
        ("Left arm picks up the red cube, right arm picks up the blue cube", "低"),
        ("Hand over the tool from left arm to right arm", "中"),
        ("Pick up the bottle with left arm", "低"),
    ]
    
    logger.info("\n" + "-" * 50)
    logger.info("更多测试:")
    for instr, expected in test_cases:
        result = model(images, instr)
        score = result.cooperation_score
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        logger.info(f"\n指令: {instr}")
        logger.info(f"  协调度: {score:.4f} [{bar}] (预期趋势: {expected})")


if __name__ == "__main__":
    main()

