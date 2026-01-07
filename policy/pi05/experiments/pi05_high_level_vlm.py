#!/usr/bin/env python3
"""
Pi0.5 High-Level VLM Language Generator

利用 pi0.5 预训练模型中的 PaliGemma VLM 部分，将高层任务指令分解为左右臂的子任务指令。

输入:
    - prompt: 高层语言指令 (如 "Grab the red block with left arm and pass to right arm")
    - image: 一张工作区图片 (RGB, 224x224)

输出:
    - left_arm_prompt: 左臂任务指令
    - right_arm_prompt: 右臂任务指令

支持三种模式:
    1. paligemma: 使用 pi0.5 内置的 PaliGemma VLM (需要预训练权重)
    2. rule: 基于规则的分解 (用于快速测试)
    3. api: 使用外部 VLM API (如 OpenAI GPT-4V)

Usage:
    cd /data0/users/haoce/RoboTwin/policy/pi05
    
    # 使用规则模式快速测试
    python experiments/pi05_high_level_vlm.py --mode rule
    
    # 使用 PaliGemma 模式
    python experiments/pi05_high_level_vlm.py --mode paligemma --use-pretrained
    
    # 使用外部 API (需要设置环境变量 OPENAI_API_KEY)
    python experiments/pi05_high_level_vlm.py --mode api --api-provider openai
"""

import argparse
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 预训练模型 URL
PRETRAINED_MODELS = {
    "pi05_base": "s3://openpi-assets/checkpoints/pi05_base/params",
    "pi0_base": "s3://openpi-assets/checkpoints/pi0_base/params",
}


def load_image_auto(image) -> np.ndarray:
    """
    自动加载图像，支持多种输入格式
    
    Args:
        image: 可以是以下类型:
            - str: 图片文件路径
            - np.ndarray: numpy 数组 (H, W, 3)
            - PIL.Image: PIL 图像对象
            
    Returns:
        np.ndarray: RGB 图像数组 (H, W, 3)
    """
    if isinstance(image, str):
        # 图片路径
        if not Path(image).exists():
            raise FileNotFoundError(f"图片文件不存在: {image}")
        pil_image = Image.open(image).convert('RGB')
        return np.array(pil_image)
    elif isinstance(image, np.ndarray):
        return image
    elif hasattr(image, 'convert'):  # PIL Image
        return np.array(image.convert('RGB'))
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")


class HighLevelVLMBackend(ABC):
    """High-Level VLM 后端抽象基类。"""
    
    @abstractmethod
    def generate_arm_prompts(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> dict[str, str]:
        """
        生成左右臂的 prompt
        
        Args:
            image: RGB 图像 (H, W, 3) 或图片路径
            instruction: 高层任务指令
            
        Returns:
            dict: {
                "left_arm_prompt": 左臂任务指令,
                "right_arm_prompt": 右臂任务指令,
                "raw_output": 原始输出
            }
        """
        pass
    
    def __call__(self, image, instruction: str) -> dict[str, str]:
        """调用接口，自动处理图像输入。"""
        image = load_image_auto(image)
        return self.generate_arm_prompts(image, instruction)


class RuleBasedBackend(HighLevelVLMBackend):
    """基于规则的任务分解（用于快速测试）。"""
    
    def generate_arm_prompts(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> dict[str, str]:
        """使用规则分解任务。"""
        import re
        
        instruction_lower = instruction.lower()
        
        left_prompt = ""
        right_prompt = ""
        
        # 传递/交接任务
        if any(word in instruction_lower for word in ["handover", "pass", "transfer", "hand"]):
            if "left" in instruction_lower and "right" in instruction_lower:
                left_prompt = "Grab the object and move to handover position"
                right_prompt = "Receive the object from left arm and place at target"
            else:
                left_prompt = "Pick up the object"
                right_prompt = "Receive and place the object"
                
        # 协同抓取
        elif "together" in instruction_lower or "cooperative" in instruction_lower:
            left_prompt = f"Left arm assist: {instruction}"
            right_prompt = f"Right arm lead: {instruction}"
            
        # 双臂独立任务 - 明确指定了 left arm 和 right arm
        elif "left arm" in instruction_lower and "right arm" in instruction_lower:
            # 尝试多种分割方式
            split_patterns = [", and then ", " and then ", ", and ", " and "]
            parts = [instruction]
            
            for pattern in split_patterns:
                if pattern in instruction.lower():
                    parts = re.split(pattern, instruction, flags=re.IGNORECASE)
                    break
            
            for part in parts:
                part_lower = part.lower()
                cleaned_part = part.strip()
                
                if "left arm" in part_lower:
                    left_prompt = cleaned_part
                elif "right arm" in part_lower:
                    right_prompt = cleaned_part
        
        # 序列任务 - 使用 "and then" 或 "other arm" 分割
        elif any(pattern in instruction_lower for pattern in [" and then ", "other arm", "another arm"]):
            # 尝试按 "and then" 分割
            split_patterns = [", and then ", " and then ", ", then "]
            parts = [instruction]
            
            for pattern in split_patterns:
                if pattern in instruction.lower():
                    parts = re.split(pattern, instruction, flags=re.IGNORECASE)
                    break
            
            if len(parts) >= 2:
                # 第一个任务给左臂，第二个任务给右臂
                left_prompt = parts[0].strip()
                right_prompt = parts[1].strip()
                # 清理 "use the other arm to" 这样的前缀
                right_prompt = re.sub(r'^use\s+(the\s+)?(other|another)\s+arm\s+to\s+', '', right_prompt, flags=re.IGNORECASE)
            else:
                # 无法分割，默认处理
                left_prompt = instruction
                right_prompt = "Coordinate with left arm"
                    
        # 单臂任务（另一臂准备）
        elif "left arm" in instruction_lower:
            left_prompt = instruction
            right_prompt = "Stand by and prepare to assist"
        elif "right arm" in instruction_lower:
            left_prompt = "Stand by and prepare to assist"
            right_prompt = instruction
            
        # 默认情况 - 假设是左臂为主
        else:
            left_prompt = instruction
            right_prompt = "Coordinate with left arm"
        
        return {
            "left_arm_prompt": left_prompt,
            "right_arm_prompt": right_prompt,
            "raw_output": f"[Rule-based decomposition]\nLeft: {left_prompt}\nRight: {right_prompt}",
        }


class PaliGemmaBackend(HighLevelVLMBackend):
    """
    使用 Pi0.5 内置的 PaliGemma VLM 进行任务分解。
    
    Note: Pi0.5 的 PaliGemma 主要用于理解视觉输入并生成动作，
    文本生成能力有限，因此会结合规则方法。
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
        print(tokenizer_path)
        with tokenizer_path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        
        # 加载模型
        logger.info("加载 Pi0.5 VLM 模型...")
        self._load_vlm_model(pi0_config, _gemma, _siglip, _model, nnx, nnx_bridge)
        logger.info("模型加载完成!")
        
        # 规则后端作为后备
        self._rule_backend = RuleBasedBackend()
        
    def _load_vlm_model(self, pi0_config, _gemma, _siglip, _model, nnx, nnx_bridge):
        """加载 VLM 模型组件。"""
        config = pi0_config.Pi0Config(pi05=True)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        
        rngs = nnx.Rngs(self.jax.random.key(0))
        
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
        fake_image = self.jnp.ones((1, 224, 224, 3), dtype=self.jnp.float32)
        self.img_encoder.lazy_init(fake_image, train=False, rngs=rngs)
        
        # 加载权重
        if self.checkpoint_path:
            logger.info(f"从 {self.checkpoint_path} 加载权重...")
            params = _model.restore_params(self.checkpoint_path)
            if 'PaliGemma' in params:
                logger.info("加载 PaliGemma 权重...")
            
        self.config = config
        self.paligemma_config = paligemma_config
    
    def preprocess_image(self, image: np.ndarray):
        """预处理图像。"""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.shape[:2] != (224, 224):
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((224, 224), Image.BILINEAR)
            image = np.array(pil_image).astype(np.float32) / 255.0
        
        image = image * 2.0 - 1.0
        image = image[np.newaxis, ...]
        
        return self.jnp.array(image)
    
    def generate_arm_prompts(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> dict[str, str]:
        """
        使用 PaliGemma 生成 arm prompts。
        
        由于 pi0.5 主要用于动作生成，这里结合图像特征和规则方法。
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image)
        
        # 编码图像 (获取视觉特征)
        image_tokens, _ = self.img_encoder(image_tensor, train=False)
        
        # 使用规则方法生成基础分解
        result = self._rule_backend.generate_arm_prompts(image, instruction)
        result["raw_output"] = f"[PaliGemma + Rule-based]\n{result['raw_output']}"
        
        return result


class OpenAIVLMBackend(HighLevelVLMBackend):
    """使用 OpenAI GPT-4V API 进行任务分解。"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-vision-preview"):
        """
        初始化 OpenAI 后端
        
        Args:
            api_key: OpenAI API Key (也可从环境变量 OPENAI_API_KEY 获取)
            model: 模型名称
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("未设置 OPENAI_API_KEY，将使用规则方法作为后备")
            self._fallback = RuleBasedBackend()
        else:
            self._fallback = None
            
    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为 base64。"""
        import base64
        import io
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def generate_arm_prompts(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> dict[str, str]:
        """使用 GPT-4V 生成 arm prompts。"""
        if self._fallback:
            return self._fallback.generate_arm_prompts(image, instruction)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # 编码图像
            image_base64 = self._encode_image(image)
            
            # 构建 prompt
            prompt = f"""Look at this image of a bimanual robot workspace.

Task: {instruction}

Please decompose this task into two separate subtasks - one for the left arm and one for the right arm.

Format your response EXACTLY as:
Left Arm: [specific subtask for left arm]
Right Arm: [specific subtask for right arm]

Be specific and concise."""

            # 调用 API
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=256,
            )
            
            raw_output = response.choices[0].message.content
            
            # 解析响应
            return self._parse_response(raw_output)
            
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return RuleBasedBackend().generate_arm_prompts(image, instruction)
    
    def _parse_response(self, response: str) -> dict[str, str]:
        """解析 API 响应。"""
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
            "raw_output": response,
        }


class Pi05HighLevelVLM:
    """
    Pi0.5 High-Level VLM 统一接口
    
    支持多种后端模式:
    - rule: 基于规则的分解
    - paligemma: 使用 Pi0.5 内置的 PaliGemma
    - api: 使用外部 VLM API (如 OpenAI)
    """
    
    def __init__(
        self,
        mode: str = "rule",
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = False,
        api_key: Optional[str] = None,
        api_provider: str = "openai",
    ):
        """
        初始化 Pi0.5 High-Level VLM
        
        Args:
            mode: 后端模式 ("rule", "paligemma", "api")
            checkpoint_path: 检查点路径 (paligemma 模式)
            use_pretrained: 是否使用预训练模型 (paligemma 模式)
            api_key: API Key (api 模式)
            api_provider: API 提供商 (api 模式)
        """
        self.mode = mode
        
        if mode == "rule":
            self.backend = RuleBasedBackend()
        elif mode == "paligemma":
            self.backend = PaliGemmaBackend(
                checkpoint_path=checkpoint_path,
                use_pretrained=use_pretrained,
            )
        elif mode == "api":
            if api_provider == "openai":
                self.backend = OpenAIVLMBackend(api_key=api_key)
            else:
                raise ValueError(f"未知的 API 提供商: {api_provider}")
        else:
            raise ValueError(f"未知的模式: {mode}")
        
        logger.info(f"初始化 High-Level VLM, 模式: {mode}")
    
    def generate_arm_prompts(
        self,
        image,
        instruction: str,
    ) -> dict[str, str]:
        """
        生成左右臂的 prompt
        
        Args:
            image: RGB 图像，支持以下格式:
                - str: 图片文件路径
                - np.ndarray: numpy 数组 (H, W, 3)
                - PIL.Image: PIL 图像对象
            instruction: 高层任务指令
            
        Returns:
            dict: {
                "left_arm_prompt": 左臂任务指令,
                "right_arm_prompt": 右臂任务指令,
                "raw_output": 原始输出
            }
        """
        # 自动加载图像
        image = load_image_auto(image)
        return self.backend.generate_arm_prompts(image, instruction)
    
    def __call__(
        self,
        image,
        instruction: str,
    ) -> dict[str, str]:
        """调用接口。"""
        return self.generate_arm_prompts(image, instruction)


def create_dummy_image(size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """创建测试图像。"""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def load_image(image_path: str) -> np.ndarray:
    """加载图像文件。"""
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = pil_image.resize((224, 224), Image.BILINEAR)
    return np.array(pil_image)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Pi0.5 High-Level VLM Language Generator")
    parser.add_argument("--mode", type=str, default="rule",
                        choices=["rule", "paligemma", "api"],
                        help="后端模式: rule (规则), paligemma (Pi0.5 VLM), api (外部 API)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径 (paligemma 模式)")
    parser.add_argument("--use-pretrained", action="store_true",
                        help="使用预训练的 pi0.5 模型 (paligemma 模式)")
    parser.add_argument("--api-provider", type=str, default="openai",
                        choices=["openai"],
                        help="API 提供商 (api 模式)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (api 模式，也可设置 OPENAI_API_KEY 环境变量)")
    parser.add_argument("--image", type=str, default=None,
                        help="输入图像路径（可选，默认使用测试图像）")
    parser.add_argument("--instruction", type=str,
                        default="Grab the red block with the left arm and pass it to the right arm",
                        help="高层任务指令")
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == "paligemma" and not args.use_pretrained and not args.checkpoint:
        logger.error("paligemma 模式需要指定 --use-pretrained 或 --checkpoint")
        return
    
    logger.info("=" * 60)
    logger.info("Pi0.5 High-Level VLM Language Generator")
    logger.info(f"模式: {args.mode}")
    logger.info("=" * 60)
    
    # 创建模型
    vlm = Pi05HighLevelVLM(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        use_pretrained=args.use_pretrained,
        api_key=args.api_key,
        api_provider=args.api_provider,
    )
    
    # 准备图像
    if args.image and Path(args.image).exists():
        logger.info(f"加载图像: {args.image}")
        image = load_image(args.image)
    else:
        logger.info("使用测试图像...")
        image = create_dummy_image()
    
    # 运行任务分解
    logger.info(f"\n输入指令: {args.instruction}")
    logger.info("-" * 60)
    
    result = vlm(image, args.instruction)
    
    logger.info("\n生成结果:")
    logger.info(f"  左臂指令: {result['left_arm_prompt']}")
    logger.info(f"  右臂指令: {result['right_arm_prompt']}")
    logger.info("-" * 60)
    logger.info(f"原始输出:\n{result['raw_output']}")
    
    # 测试更多指令
    test_instructions = [
        "Pick up the bottle with left arm and shake it",
        "Use both arms to lift the heavy box together",
        "Left arm grabs the red cube, right arm grabs the blue cube",
        "Hand over the tool from left arm to right arm",
        "Coordinate both arms to fold the cloth",
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("更多测试指令:")
    logger.info("=" * 60)
    
    for instr in test_instructions:
        result = vlm(image, instr)
        logger.info(f"\n指令: {instr}")
        logger.info(f"  左臂: {result['left_arm_prompt']}")
        logger.info(f"  右臂: {result['right_arm_prompt']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("完成!")
    logger.info("=" * 60)
    
    return result


# 便捷的工厂函数
def create_high_level_vlm(
    mode: str = "rule",
    checkpoint_path: Optional[str] = None,
    use_pretrained: bool = False,
    api_key: Optional[str] = None,
) -> Pi05HighLevelVLM:
    """
    创建 High-Level VLM 实例
    
    Args:
        mode: 后端模式 ("rule", "paligemma", "api")
        checkpoint_path: 检查点路径
        use_pretrained: 是否使用预训练模型
        api_key: API Key
        
    Returns:
        Pi05HighLevelVLM 实例
        
    Example:
        >>> vlm = create_high_level_vlm(mode="rule")
        >>> result = vlm(image, "Pick up the red block")
        >>> print(result["left_arm_prompt"])
        >>> print(result["right_arm_prompt"])
    """
    return Pi05HighLevelVLM(
        mode=mode,
        checkpoint_path=checkpoint_path,
        use_pretrained=use_pretrained,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()

