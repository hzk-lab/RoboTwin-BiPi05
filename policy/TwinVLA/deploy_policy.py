"""
TwinVLA Deploy Policy

RoboTwin 标准部署接口实现。
基于 TwinPi0 模型，使用 Joint Attention 让左右臂在 Transformer 层互相交流。

核心特点:
- 真正的 Joint Attention: 不是独立推理+拼接，而是在 attention 层让左右臂 tokens 互相可见
- 共享 Vision Encoder: ego view + language 共享编码
- 分离 Arm Inputs: 左右腕相机、本体感受各自独立

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

from src.twinvla_model import TwinVLAPolicy


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
    加载 TwinVLA 模型。
    
    Args:
        usr_args: 用户参数字典，来自 deploy_policy.yml 和 eval.sh
            - train_config_name: openpi 训练配置名称
            - model_name: 任务模型名称 (用于构建 checkpoint 路径)
            - checkpoint_id: checkpoint step 号
            - pi0_step: 每次推理输出的动作步数
    
    Returns:
        TwinVLAPolicy 实例
    """
    train_config_name = usr_args["train_config_name"]
    model_name = usr_args.get("model_name")
    checkpoint_id = usr_args.get("checkpoint_id")
    pi0_step = usr_args["pi0_step"]
    
    # 优先使用显式传入的 checkpoint_dir；否则按旧规则拼路径
    checkpoint_dir = usr_args.get("checkpoint_dir")
    if checkpoint_dir is None:
        checkpoint_dir = (
            f"/data0/users/haoce/pi0_checkpoints/"
            f"{train_config_name}/{model_name}/{checkpoint_id}"
        )
    
    print(f"[TwinVLA] Loading from checkpoint: {checkpoint_dir}")
    print(f"[TwinVLA] Config: {train_config_name}, pi0_step: {pi0_step}")
    
    model = TwinVLAPolicy(
        train_config_name=train_config_name,
        checkpoint_dir=checkpoint_dir,
        pi0_step=pi0_step,
    )
    
    return model


def eval(TASK_ENV, model, observation):
    """
    执行单步评估。
    
    遵循 RoboTwin 的标准接口：
    1. 处理观测
    2. 获取动作
    3. 执行动作并更新观测
    
    Args:
        TASK_ENV: RoboTwin 任务环境
        model: TwinVLA Policy
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
        model: TwinVLA Policy
    """
    model.reset_obsrvationwindows()
