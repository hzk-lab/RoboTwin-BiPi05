"""
TwinVLA1 Deploy Policy - 双VLA + Joint Attention

RoboTwin 标准部署接口实现。
基于两个独立的 Pi0 模型，分别处理左右臂，通过 joint attention 实现跨臂协调。

核心特点:
- 两个完全独立的 VLA 模型 (各自有完整的 vision/language encoder)
- 左臂 VLA: 输出32维，只有前7维有效
- 右臂 VLA: 输出32维，只有7-13维有效
- 特定的噪声添加策略 (只对有效维度做 flow matching)
- 可选的跨 VLA joint attention

Usage:
    在 eval.sh 中指定 checkpoint 路径，然后运行评估脚本。
"""

import numpy as np
import os
import sys

# 添加路径
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from src.dual_vla import DualVLAPolicy, create_dual_vla_policy


def encode_obs(observation):
    """
    预处理观测数据，转换为模型需要的格式。
    
    Args:
        observation: RoboTwin 环境返回的原始观测
            - observation/head_camera/rgb: 前置相机图像
            - observation/right_camera/rgb: 右腕相机图像
            - observation/left_camera/rgb: 左腕相机图像
            - joint_action/vector: 机器人状态 [16]
    
    Returns:
        input_rgb_arr: [front, right_wrist, left_wrist] 图像列表
        input_state: 状态向量 [16]
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    
    return input_rgb_arr, input_state


def get_model(usr_args):
    """
    加载 TwinVLA1 (Dual VLA) 模型。
    
    Args:
        usr_args: 用户参数字典，来自 deploy_policy.yml 和 eval.sh
            - train_config_name: openpi 训练配置名称
            - left_model_name: 左臂模型名称
            - right_model_name: 右臂模型名称
            - left_checkpoint_id: 左臂 checkpoint step 号
            - right_checkpoint_id: 右臂 checkpoint step 号
            - pi0_step: 每次推理输出的动作步数
    
    Returns:
        DualVLAPolicy 实例
    """
    train_config_name = usr_args["train_config_name"]
    pi0_step = usr_args["pi0_step"]
    
    # 左臂 checkpoint
    left_checkpoint_dir = usr_args.get("left_checkpoint_dir")
    if left_checkpoint_dir is None:
        left_model_name = usr_args.get("left_model_name", usr_args.get("model_name"))
        left_checkpoint_id = usr_args.get("left_checkpoint_id", usr_args.get("checkpoint_id"))
        # 优先查找 TwinVLA1 的 checkpoints，然后查找 pi0 的 checkpoints
        twinvla1_ckpt = (
            f"/data0/users/haoce/RoboTwin/policy/TwinVLA1/checkpoints/"
            f"{train_config_name}/{left_model_name}/{left_checkpoint_id}"
        )
        pi0_ckpt = (
            f"/data0/users/haoce/RoboTwin/policy/pi0/checkpoints/"
            f"{train_config_name}/{left_model_name}/{left_checkpoint_id}"
        )
        left_checkpoint_dir = twinvla1_ckpt if os.path.exists(twinvla1_ckpt) else pi0_ckpt
    
    # 右臂 checkpoint
    right_checkpoint_dir = usr_args.get("right_checkpoint_dir")
    if right_checkpoint_dir is None:
        right_model_name = usr_args.get("right_model_name", usr_args.get("model_name"))
        right_checkpoint_id = usr_args.get("right_checkpoint_id", usr_args.get("checkpoint_id"))
        # 优先查找 TwinVLA1 的 checkpoints，然后查找 pi0 的 checkpoints
        twinvla1_ckpt = (
            f"/data0/users/haoce/RoboTwin/policy/TwinVLA1/checkpoints/"
            f"{train_config_name}/{right_model_name}/{right_checkpoint_id}"
        )
        pi0_ckpt = (
            f"/data0/users/haoce/RoboTwin/policy/pi0/checkpoints/"
            f"{train_config_name}/{right_model_name}/{right_checkpoint_id}"
        )
        right_checkpoint_dir = twinvla1_ckpt if os.path.exists(twinvla1_ckpt) else pi0_ckpt
    
    print(f"[TwinVLA1] Loading Dual VLA")
    print(f"[TwinVLA1] Left checkpoint: {left_checkpoint_dir}")
    print(f"[TwinVLA1] Right checkpoint: {right_checkpoint_dir}")
    print(f"[TwinVLA1] Config: {train_config_name}, pi0_step: {pi0_step}")
    
    model = create_dual_vla_policy(
        train_config_name=train_config_name,
        left_checkpoint_dir=left_checkpoint_dir,
        right_checkpoint_dir=right_checkpoint_dir,
        pi0_step=pi0_step,
    )
    
    return model


def eval(TASK_ENV, model, observation):
    """
    执行单步评估。
    
    遵循 RoboTwin 的标准接口：
    1. 处理观测
    2. 获取动作 (通过 Dual VLA + Joint Attention)
    3. 执行动作并更新观测
    
    Args:
        TASK_ENV: RoboTwin 任务环境
        model: DualVLA Policy
        observation: 当前观测
    """
    # 首次调用时设置语言指令
    if model.instruction is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
    
    # 处理观测
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    
    # 获取动作序列
    actions = model.get_action()[:model.pi0_step]
    
    # 执行每一步动作
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):  
    """
    重置模型状态。
    
    在每个评估 episode 开始时调用，清空状态。
    
    Args:
        model: DualVLA Policy
    """
    model.reset_obsrvationwindows()
