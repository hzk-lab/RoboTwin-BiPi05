#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import os
import sys
import time
# 优先使用当前策略目录下的本地 openpi 代码，避免导入到系统已安装的旧版本
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        # checkpoint 位置：
        # data0 本地检查点结构示例：
        # /data0/users/haoce/RoboTwin/policy/pi0/checkpoints/pi0_base_aloha_robotwin_lora/shake_bottle-demo_clean-50/15000
        ckpt_dir = f"/c20250502/zxr/checkpoints/pi0_base_aloha_robotwin_lora/pi0/{self.checkpoint_id}"
        self.policy = _policy_config.create_trained_policy(
            config,
            ckpt_dir,
            robotwin_repo_id=model_name)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        
        # Inference time tracking
        self.inference_times = []
        self.warmup_done = False
        self.measure_time = True  # 设置为 True 来启用时间测量

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        
        if self.measure_time:
            # JAX 需要先 block_until_ready 确保计算完成
            start_time = time.perf_counter()
            actions = self.policy.infer(self.observation_window)["actions"]
            # 确保 JAX 计算完成（block_until_ready）
            if hasattr(actions, 'block_until_ready'):
                actions.block_until_ready()
            else:
                # numpy array，不需要 block
                pass
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            
            if not self.warmup_done:
                # 第一次调用是 warmup（包含 JIT 编译时间）
                print(f"[Warmup] Inference time (with JIT compilation): {inference_time_ms:.2f} ms")
                self.warmup_done = True
            else:
                self.inference_times.append(inference_time_ms)
                print(f"[Inference] Time: {inference_time_ms:.2f} ms")
            
            return actions
        else:
            return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
    
    def print_inference_stats(self):
        """打印 inference time 统计信息"""
        if len(self.inference_times) == 0:
            print("No inference time data collected (excluding warmup)")
            return
        
        times = np.array(self.inference_times)
        print("\n" + "="*50)
        print("PI0 High-Level Inference Time Statistics")
        print("="*50)
        print(f"  Total inferences: {len(times)}")
        print(f"  Mean:   {np.mean(times):.2f} ms")
        print(f"  Std:    {np.std(times):.2f} ms")
        print(f"  Min:    {np.min(times):.2f} ms")
        print(f"  Max:    {np.max(times):.2f} ms")
        print(f"  Median: {np.median(times):.2f} ms")
        print(f"  FPS:    {1000 / np.mean(times):.2f}")
        print("="*50 + "\n")
