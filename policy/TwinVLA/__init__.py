"""
TwinVLA - Twin Vision-Language-Action Model for Bimanual Robot Control

基于 JAX 原生方案，将两个预训练的单臂 Pi0 模型组合，
实现从单臂技能到双臂协调的泛化。

核心思想:
- 加载两个 openpi Policy 实例（左臂和右臂）
- 分割观测、分别推理、合并动作
"""

from .deploy_policy import *
