"""
TwinVLA1 - Dual VLA with Joint Attention

双VLA架构实现，使用独立的左右臂VLA并通过causal joint attention进行信息交流。

架构特点:
- 两个完全独立的 VLA 模型 (各自有完整的 vision/language encoder)
- 左臂 VLA: 输出32维，只有前7维有效
- 右臂 VLA: 输出32维，只有7-13维有效
- 特定的噪声添加策略 (只对有效维度做 flow matching)
- Causal joint attention 实现跨臂信息交流
"""

from .deploy_policy import encode_obs, get_model, eval, reset_model
