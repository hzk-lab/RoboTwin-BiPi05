export BIG=/vepfs-mlp2/mlp-public/haoce/zxr/.cache_big

# 兜底：任何写 ~/.cache 的都进 BIG
mkdir -p $BIG/root_cache
[ -L /root/.cache ] || (rm -rf /root/.cache && ln -sfn $BIG/root_cache /root/.cache)

# HF / transformers / datasets
export HF_HOME=$BIG/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# wandb
export WANDB_DIR=$BIG/wandb
export WANDB_CACHE_DIR=$BIG/wandb_cache

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR"
