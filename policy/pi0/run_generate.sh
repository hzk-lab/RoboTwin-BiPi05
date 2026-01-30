#!/usr/bin/env bash
set -e
export LD_LIBRARY_PATH=/vepfs-mlp2/mlp-public/haoce/conda_env/RoboTwin/lib:${LD_LIBRARY_PATH:-}
bash generate.sh "$@"
