#!/usr/bin/env python3
"""
测试修改后的 compute_loss (多模态文本生成)

使用方式:
    # 使用本地权重 + 真实图片
    PYTHONPATH=../src python test_compute_loss.py \
        --local-weights /home/users/xuanran/.cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz \
        --image ~/RoboTwin-BiPi05/mid2.jpg \
        --instruction "Pick up the rack"
    
    # 不指定图片 (使用随机图片)
    PYTHONPATH=../src python test_compute_loss.py \
        --local-weights /home/users/xuanran/.cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz \
        --instruction "Grasp the bottle and place it on the table"
"""

import argparse
import os
import sys
import logging

# 添加 src 目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), "src")
sys.path.insert(0, src_dir)

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import flax.traverse_util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_local_paligemma_weights(weights_path: str, params: dict) -> dict:
    """从本地加载 PaliGemma 权重 (支持 .npz 格式)"""
    logger.info(f"从本地加载 PaliGemma 权重: {weights_path}")
    
    if weights_path.endswith('.npz'):
        # 加载 npz 格式 (与 Pi0 兼容)
        with open(weights_path, 'rb') as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        
        logger.info(f"加载了 {len(flat_params)} 个参数")
        
        # 转换为嵌套字典格式
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        
        # 合并参数
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        
        result = {}
        for k, v in flat_loaded.items():
            if k in flat_ref:
                result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        
        # 添加缺失的参数 (如 action expert)
        for k in flat_ref:
            if k not in result:
                result[k] = flat_ref[k]
        
        logger.info(f"合并后参数数量: {len(result)}")
        return flax.traverse_util.unflatten_dict(result, sep="/")
    else:
        raise ValueError(f"不支持的权重格式: {weights_path}，请使用 .npz 文件")


def main():
    parser = argparse.ArgumentParser(description="测试 compute_loss (多模态文本生成)")
    parser.add_argument("--local-weights", type=str, default=None,
                        help="本地 PaliGemma 权重路径 (.npz 格式)")
    parser.add_argument("--image", type=str, default=None,
                        help="输入图片路径")
    parser.add_argument("--instruction", type=str, 
                        default="Pick up the red block and place it on the plate",
                        help="任务指令")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("测试 compute_loss (文本生成)")
    logger.info("=" * 60)
    
    # 1. 创建模型
    from openpi.models import pi0_config
    from openpi.models.pi0 import Pi0
    from openpi.models.model import Observation
    from openpi.training.weight_loaders import PaliGemmaWeightLoader
    
    logger.info("创建模型配置...")
    config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
    )
    
    logger.info("创建 Pi0 模型...")
    rngs = nnx.Rngs(0)
    model = Pi0(config, rngs)
    
    # 2. 加载 PaliGemma 权重
    logger.info("加载 PaliGemma 预训练权重...")
    graphdef, state = nnx.split(model)
    params = state.to_pure_dict()
    
    if args.local_weights and os.path.exists(args.local_weights):
        # 使用本地权重
        params = load_local_paligemma_weights(args.local_weights, params)
    else:
        # 使用 GCS 下载的权重 (npz 格式，与 Pi0 兼容)
        logger.info("使用 GCS 权重: gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz")
        loader = PaliGemmaWeightLoader()
        params = loader.load(params)
    
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    logger.info("权重加载完成!")
    
    # 3. 准备真实任务输入
    from PIL import Image
    from openpi.models import tokenizer as _tokenizer
    from openpi.shared import image_tools
    from openpi.models import model as _model
    
    # 真实的任务指令
    task_instruction = args.instruction
    
    # 加载真实图片 (如果有的话) 或使用测试图片
    image_path = args.image
    
    if image_path and os.path.exists(image_path):
        logger.info(f"加载图片: {image_path}")
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        img = img / 255.0 * 2.0 - 1.0  # 归一化到 [-1, 1]
        img = jnp.asarray(img)
        img = image_tools.resize_with_pad(img, 224, 224)
    else:
        # 使用随机测试图片
        logger.info("未提供图片，使用随机测试图片")
        img = np.random.rand(224, 224, 3).astype(np.float32) * 2.0 - 1.0
        img = jnp.asarray(img)
    
    # 创建 3 个视角的图像输入
    images = {
        "base_0_rgb": img[None, ...],
        "left_wrist_0_rgb": img[None, ...],
        "right_wrist_0_rgb": img[None, ...],
    }
    image_masks = {
        "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
    }
    
    # Tokenize 任务指令
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=48)
    tokens, token_mask = tokenizer.tokenize(task_instruction)
    
    logger.info(f"任务指令: {task_instruction}")
    logger.info(f"Token 数量: {int(np.sum(token_mask))}")
    
    # 创建真实的 Observation
    real_obs = Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((1, 14), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(tokens)[None, :],
        tokenized_prompt_mask=jnp.asarray(token_mask)[None, :],
    )
    
    dummy_actions = jnp.zeros((1, 50, 14), dtype=jnp.float32)
    
    # 4. 调用 compute_loss (多模态文本生成)
    logger.info("-" * 60)
    logger.info("调用 compute_loss (多模态文本生成)...")
    logger.info("-" * 60)
    
    rng = jax.random.key(0)
    output = model.compute_loss(rng, real_obs, dummy_actions, train=False)
    
    logger.info("-" * 60)
    logger.info(f"输出 shape: {output.shape}")
    logger.info(f"生成的 token IDs: {output}")
    
    # 解码输出
    decoded_text = tokenizer._tokenizer.decode(output[0].tolist())
    logger.info(f"解码文本: {decoded_text}")


if __name__ == "__main__":
    main()


