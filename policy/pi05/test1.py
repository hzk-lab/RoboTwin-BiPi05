from experiments.pi05_high_level_vlm import create_high_level_vlm
import numpy as np

# 创建 VLM 实例
vlm = create_high_level_vlm(mode="paligemma")  # 或 "paligemma", "api"

# 准备输入
image = "/data0/users/haoce/Screenshot_1.png"
# instruction = "Put the cup over the black mark, and then use the other arm to use the stick to stir it."
instruction = "Take the red block from the left arm and put it on the blue pad on the right side."

# 生成分解结果
result = vlm(image, instruction)
print(f"{result['left_arm_prompt']}")
print(f"{result['right_arm_prompt']}")