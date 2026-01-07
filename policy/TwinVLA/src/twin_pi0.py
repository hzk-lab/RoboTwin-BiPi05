"""
TwinPi0 - 双臂 Pi0 模型，实现真正的 Joint Attention

核心改动：
1. 将每个时间步的动作拆成两个 token (L_t, R_t)
2. 序列从 H 个 token 变成 2H 个 token: [L0, R0, L1, R1, ...]
3. 使用零填充保持 action_in_proj 和 action_out_proj 权重兼容
4. 新的 attention mask 让左右臂能互相看到

Token 布局:
原始: [T0, T1, ..., T_{H-1}]  每个 T_i = [left_8, right_8]
TwinPi0: [L0, R0, L1, R1, ..., L_{H-1}, R_{H-1}]
  - L_t = [left_8, zeros_8]
  - R_t = [zeros_8, right_8]

Attention 规则:
- L_t 可看: Common + L_{<=t} + R_{<t}
- R_t 可看: Common + R_{<=t} + L_{<=t}
"""

import logging
from pathlib import Path

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pi05" / "src"))

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)


def make_bimanual_ar_mask(action_horizon: int) -> jnp.ndarray:
    """
    创建 2H 长度的双臂 AR mask。
    
    对于 2H 个 action tokens (索引 0 到 2H-1):
    - 偶数索引 2t = L_t (左臂时间步 t)
    - 奇数索引 2t+1 = R_t (右臂时间步 t)
    
    Mask 规则:
    - L_t (idx=2t) 可 attend: L_{0..t} + R_{0..t-1}
    - R_t (idx=2t+1) 可 attend: R_{0..t} + L_{0..t}
    
    Args:
        action_horizon: 原始时间步数 H
    
    Returns:
        mask: [2H, 2H] bool array
    """
    n = 2 * action_horizon
    
    # 使用向量化操作构建 mask
    i = jnp.arange(n)
    j = jnp.arange(n)
    
    # query 和 key 的时间步
    t_i = i // 2  # [n]
    t_j = j // 2  # [n]
    
    # 是否是左臂
    is_left_i = (i % 2 == 0)  # [n]
    is_left_j = (j % 2 == 0)  # [n]
    
    # 广播到 [n, n]
    t_i = t_i[:, None]  # [n, 1]
    t_j = t_j[None, :]  # [1, n]
    is_left_i = is_left_i[:, None]  # [n, 1]
    is_left_j = is_left_j[None, :]  # [1, n]
    
    # L_t 可看: L_{<=t} + R_{<t}
    left_query_mask = (
        (is_left_j & (t_j <= t_i)) |  # L_t 看 L_{<=t}
        (~is_left_j & (t_j < t_i))     # L_t 看 R_{<t}
    )
    
    # R_t 可看: R_{<=t} + L_{<=t}
    right_query_mask = (
        (~is_left_j & (t_j <= t_i)) |  # R_t 看 R_{<=t}
        (is_left_j & (t_j <= t_i))      # R_t 看 L_{<=t}
    )
    
    # 根据 query 是左臂还是右臂选择 mask
    mask = jnp.where(is_left_i, left_query_mask, right_query_mask)
    
    return mask


def interleave_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """
    将 [B, H, D] 的双臂动作转换为 [B, 2H, D] 的 interleaved 格式。
    
    输入: actions[..., :D/2] 是左臂, actions[..., D/2:] 是右臂
    输出: [L0, R0, L1, R1, ...] 其中 L_t = [left, zeros], R_t = [zeros, right]
    """
    batch_size, horizon, action_dim = actions.shape
    half_dim = action_dim // 2
    
    # 拆分左右臂
    left_actions = actions[..., :half_dim]   # [B, H, D/2]
    right_actions = actions[..., half_dim:]  # [B, H, D/2]
    
    # 零填充
    zeros = jnp.zeros_like(left_actions)
    left_padded = jnp.concatenate([left_actions, zeros], axis=-1)   # [B, H, D]: [left, 0]
    right_padded = jnp.concatenate([zeros, right_actions], axis=-1) # [B, H, D]: [0, right]
    
    # Interleave: [L0, R0, L1, R1, ...]
    # Stack: [B, H, 2, D] then reshape to [B, 2H, D]
    interleaved = jnp.stack([left_padded, right_padded], axis=2)  # [B, H, 2, D]
    interleaved = interleaved.reshape(batch_size, 2 * horizon, action_dim)  # [B, 2H, D]
    
    return interleaved


def restore_actions(interleaved_output: jnp.ndarray) -> jnp.ndarray:
    """
    将 [B, 2H, D] 的 interleaved 输出还原为 [B, H, D] 的双臂动作。
    
    输入: [L0, R0, L1, R1, ...] 其中 L_t 输出前半维有效, R_t 输出后半维有效
    输出: [B, H, D] 其中 [..., :D/2] 是左臂, [..., D/2:] 是右臂
    """
    batch_size = interleaved_output.shape[0]
    double_horizon = interleaved_output.shape[1]
    action_dim = interleaved_output.shape[2]
    
    horizon = double_horizon // 2
    half_dim = action_dim // 2
    
    # 偶数位置取前半维 (左臂)
    left_output = interleaved_output[:, 0::2, :half_dim]  # [B, H, D/2]
    # 奇数位置取后半维 (右臂)
    right_output = interleaved_output[:, 1::2, half_dim:]  # [B, H, D/2]
    
    # 合并
    restored = jnp.concatenate([left_output, right_output], axis=-1)  # [B, H, D]
    
    return restored


class TwinPi0(_pi0.Pi0):
    """
    TwinPi0 - 带 Joint Attention 的 Pi0 模型
    
    核心改动:
    1. embed_suffix: 将 [B, H, D] 转换为 [B, 2H, D] 的 interleaved tokens
    2. sample_actions: 使用新的 attention mask 并还原输出
    
    不改变模型权重结构，可以直接加载 Pi0 checkpoint。
    """
    
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        logger.info(f"TwinPi0 initialized with Joint Attention (horizon={config.action_horizon})")
    
    @override
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """
        嵌入 suffix tokens，将动作转换为 interleaved 格式。
        
        输入 noisy_actions: [B, H, D]
        输出 tokens: [B, 2H, emb] (interleaved: [L0, R0, L1, R1, ...])
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        
        # 将动作转换为 interleaved 格式: [B, H, D] -> [B, 2H, D]
        interleaved_actions = interleave_actions(noisy_actions)
        
        # 投影到 embedding 空间
        action_tokens = self.action_in_proj(interleaved_actions)
        
        # timestep embedding
        time_emb = _pi0.posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # 注意: 现在是 2H 个 tokens
            double_horizon = 2 * self.action_horizon
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=double_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        
        tokens.append(action_expert_tokens)
        
        # input_mask: 2H 个 tokens 都是有效的
        double_horizon = 2 * self.action_horizon
        input_mask.append(jnp.ones((obs.state.shape[0], double_horizon), dtype=jnp.bool_))
        
        # ar_mask: 第一个 action token 是新块，后面根据 bimanual mask 处理
        # 简化: 全部标记为 True (新块)，让 bimanual mask 处理细节
        ar_mask += [True] * double_horizon
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        采样 actions，使用 Joint Attention 让左右臂协调。
        
        关键改动:
        1. noise 仍是 [B, H, D]，在 embed_suffix 中转换为 [B, 2H, D]
        2. suffix_attn_mask 使用预计算的 bimanual mask
        3. 输出从 [B, 2H, D] 还原为 [B, H, D]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        
        # noise 仍是原始形状 [B, H, D]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 首先对 prefix 做一次 forward 填充 KV cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        # 计算 bimanual attention mask (suffix 内部)
        # Shape: [2H, 2H]
        double_horizon = 2 * self.action_horizon
        bimanual_mask = make_bimanual_ar_mask(self.action_horizon)

        def step(carry):
            x_t, time = carry
            
            # embed_suffix 会将 [B, H, D] 转换为 [B, 2H, emb]
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            # suffix_attn_mask: 使用 bimanual mask
            # Shape: [B, 2H, 2H]
            suffix_attn_mask = jnp.broadcast_to(
                bimanual_mask[None, :, :], 
                (batch_size, double_horizon, double_horizon)
            )
            
            # 如果有 state token (pi0 non-pi05)，需要扩展 mask
            suffix_len = suffix_tokens.shape[1]
            if suffix_len > double_horizon:
                # 有额外的 state token
                num_extra = suffix_len - double_horizon
                # state token 可以被所有 action tokens 看到
                extra_col = jnp.ones((batch_size, double_horizon, num_extra), dtype=bool)
                # state token 不能看 action tokens
                extra_row = jnp.zeros((batch_size, num_extra, suffix_len), dtype=bool)
                # state token 自己看自己
                extra_row = extra_row.at[:, :, :num_extra].set(True)
                
                # 组装
                action_part = jnp.concatenate([extra_col, suffix_attn_mask], axis=-1)
                suffix_attn_mask = jnp.concatenate([extra_row, action_part], axis=1)
            
            # prefix_attn_mask: suffix 如何看 prefix
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            
            # 合并 mask
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            
            # positions
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            
            # 取最后 2H 个 tokens 的输出
            action_output = suffix_out[:, -double_horizon:]
            
            # 投影回 action 空间: [B, 2H, emb] -> [B, 2H, D]
            v_t_interleaved = self.action_out_proj(action_output)
            
            # 还原为 [B, H, D]
            v_t = restore_actions(v_t_interleaved)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


def create_twin_pi0_from_checkpoint(
    config: pi0_config.Pi0Config,
    checkpoint_dir: str,
) -> TwinPi0:
    """
    从 Pi0 checkpoint 创建 TwinPi0 模型。
    
    由于 TwinPi0 继承自 Pi0 且不改变权重结构，
    可以直接使用 openpi 的权重加载逻辑。
    """
    from openpi.shared import download
    
    rng = jax.random.PRNGKey(42)
    model = TwinPi0(config, rngs=nnx.Rngs(rng))
    
    # 加载权重
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    params = _model.restore_params(Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    
    # 将参数加载到模型
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    
    logger.info(f"TwinPi0 loaded from {checkpoint_dir}")
    return model
