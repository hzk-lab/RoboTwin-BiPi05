#!/bin/bash
# TwinVLA1 数据处理脚本
# 使用 pi0 的数据处理流程，生成符合 LeRobot 格式的数据
#
# 用法:
#   ./process_data.sh <task_name> <setting> <expert_data_num>
#
# 示例:
#   ./process_data.sh shake_bottle demo_clean 50

task_name=${1}
setting=${2}
expert_data_num=${3}

# 切换到 pi0 目录执行数据处理 (复用 pi0 的数据处理逻辑)
cd ../pi0

python scripts/process_data.py $task_name $setting $expert_data_num

echo "Data processing completed!"
echo "Data saved to: ../pi0/processed_data/${task_name}-${setting}-${expert_data_num}"

