#!/usr/bin/env python3
"""
测试 Pi0 模型的文本生成能力。

使用方式:
    python test_pi0_text_gen.py --checkpoint <path_to_checkpoint> --prompt "Pick up the bottle"
"""

import argparse
import logging
import os
import sys

# 添加 src 目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), "src")
sys.path.insert(0, src_dir)

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tokenizer():
    """加载 PaliGemma tokenizer"""
    from openpi.shared import download
    import sentencepiece
    
    path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", 
        gs={"token": "anon"}
    )
    with path.open("rb") as f:
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return tokenizer


def create_observation(tokenizer, prompt: str, image_path: str = None, max_token_len: int = 48):
    """创建模型输入的 Observation"""
    from openpi.models.model import Observation
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_bos=True) + tokenizer.encode("\n")
    tokens_len = len(tokens)
    
    if tokens_len < max_token_len:
        padding = [False] * (max_token_len - tokens_len)
        mask = [True] * tokens_len + padding
        tokens = tokens + padding
    else:
        tokens = tokens[:max_token_len]
        mask = [True] * max_token_len
    
    tokenized_prompt = jnp.array([tokens], dtype=jnp.int32)
    tokenized_prompt_mask = jnp.array([mask], dtype=jnp.bool_)
    
    # 创建图像 (如果没有提供，使用随机图像)
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    else:
        logger.info("使用随机测试图像")
        img_array = np.random.rand(224, 224, 3).astype(np.float32) * 2.0 - 1.0
    
    # 创建 3 个视角的图像
    images = {
        "base_0_rgb": jnp.array([img_array]),
        "left_wrist_0_rgb": jnp.array([img_array]),
        "right_wrist_0_rgb": jnp.array([img_array]),
    }
    image_masks = {
        "base_0_rgb": jnp.array([True]),
        "left_wrist_0_rgb": jnp.array([True]),
        "right_wrist_0_rgb": jnp.array([True]),
    }
    
    # State (dummy)
    state = jnp.zeros((1, 14), dtype=jnp.float32)
    
    return Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def load_model(checkpoint_path: str = None, use_paligemma_weights: bool = True):
    """加载 Pi0 模型
    
    Args:
        checkpoint_path: 完整 Pi05 检查点路径 (可选)
        use_paligemma_weights: 是否自动下载并加载 PaliGemma 预训练权重
    """
    from openpi.models import pi0_config
    from openpi.models.pi0 import Pi0
    from openpi.models.model import restore_params
    from openpi.training.weight_loaders import PaliGemmaWeightLoader, CheckpointWeightLoader
    from openpi.shared import download
    import flax.nnx as nnx
    import flax.traverse_util
    
    logger.info(f"加载模型配置...")
    
    # 使用 pi05 配置
    config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    
    logger.info(f"创建模型...")
    rngs = nnx.Rngs(0)
    model = Pi0(config, rngs)
    
    # 获取模型参数
    graphdef, state = nnx.split(model)
    params = state.to_pure_dict()
    
    if checkpoint_path:
        # 加载完整 Pi05 检查点
        logger.info(f"加载完整检查点: {checkpoint_path}")
        loader = CheckpointWeightLoader(params_path=checkpoint_path)
        params = loader.load(params)
        logger.info("检查点加载完成")
    elif use_paligemma_weights:
        # 自动下载并加载 PaliGemma 预训练权重
        logger.info("自动下载 PaliGemma 预训练权重...")
        logger.info("URL: gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz")
        loader = PaliGemmaWeightLoader()
        params = loader.load(params)
        logger.info("PaliGemma 权重加载完成!")
    else:
        logger.warning("未加载任何预训练权重，使用随机初始化")
    
    # 将参数重新加载到模型
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Pi0 文本生成测试")
    parser.add_argument("--checkpoint", type=str, default=None, help="完整 Pi05 检查点路径")
    parser.add_argument("--no-paligemma", action="store_true", help="不自动下载 PaliGemma 权重")
    parser.add_argument("--prompt", type=str, default="Task: Pick up the bottle\n\nFor a bimanual robot:\nLeft Arm:", help="输入 prompt")
    parser.add_argument("--image", type=str, default=None, help="图片路径")
    parser.add_argument("--max-tokens", type=int, default=32, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Pi0 文本生成测试")
    logger.info("=" * 60)
    
    # 1. 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = load_tokenizer()
    logger.info(f"词汇表大小: {tokenizer.vocab_size()}")
    
    # 2. 显示 prompt 的 token 化
    logger.info(f"\nPrompt: {args.prompt}")
    prompt_tokens = tokenizer.encode(args.prompt, add_bos=True)
    logger.info(f"Token IDs: {prompt_tokens}")
    logger.info(f"Token 数量: {len(prompt_tokens)}")
    
    # 3. 加载模型
    logger.info("\n加载模型...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        use_paligemma_weights=not args.no_paligemma
    )
    
    # 4. 创建 observation
    logger.info("\n创建输入...")
    obs = create_observation(tokenizer, args.prompt, args.image)
    
    # 5. 调用 generate_text
    logger.info("\n开始文本生成...")
    logger.info(f"max_new_tokens: {args.max_tokens}")
    logger.info(f"temperature: {args.temperature}")
    
    try:
        generated_ids = model.generate_text(
            obs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        logger.info(f"\n生成的 Token IDs: {generated_ids}")
        
        # 解码
        if generated_ids.size > 0:
            decoded = tokenizer.decode(generated_ids[0].tolist())
            logger.info(f"解码结果: {decoded}")
        
    except Exception as e:
        logger.error(f"生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

