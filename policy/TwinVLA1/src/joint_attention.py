"""
Joint Attention Mask for TwinVLA1 Dual VLA

实现两个独立VLA之间的causal joint attention机制。

Attention Mask 设计:
- 两个VLA各自有 prefix tokens (images + language + state) 和 action tokens
- 在 decoder 层，左右臂的 action tokens 可以互相看到，遵循因果约束

Causal Joint Attention 规则:
- L_t (左臂时间步 t) 可见: prefix + L_{<=t} + R_{<t}
- R_t (右臂时间步 t) 可见: prefix + R_{<=t} + L_{<=t}

这样实现了对称的跨臂信息流动，同时保持自回归/因果约束。
"""

import jax.numpy as jnp


def make_dual_vla_ar_mask(action_horizon: int) -> jnp.ndarray:
    """
    创建双VLA的action tokens之间的AR mask。
    
    Token布局 (2*action_horizon 个 tokens):
    - 前半部分 [0, ah): 左臂 action tokens L_0, L_1, ..., L_{ah-1}
    - 后半部分 [ah, 2ah): 右臂 action tokens R_0, R_1, ..., R_{ah-1}
    
    Attention 规则:
    - L_t 可见: L_{<=t} + R_{<t}
    - R_t 可见: R_{<=t} + L_{<=t}
    
    Args:
        action_horizon: 动作时间步数
        
    Returns:
        mask: [2*ah, 2*ah] bool array, True = 可以attend
    """
    ah = action_horizon
    n = 2 * ah
    
    # 创建索引
    i = jnp.arange(n)
    j = jnp.arange(n)
    
    # 判断是左臂还是右臂 token
    # 前 ah 个是左臂，后 ah 个是右臂
    is_left_i = i < ah  # [n]
    is_left_j = j < ah  # [n]
    
    # 获取时间步索引
    # 左臂: t = i, 右臂: t = i - ah
    t_i = jnp.where(is_left_i, i, i - ah)  # [n]
    t_j = jnp.where(is_left_j, j, j - ah)  # [n]
    
    # 广播到 [n, n]
    t_i = t_i[:, None]  # [n, 1]
    t_j = t_j[None, :]  # [1, n]
    is_left_i = is_left_i[:, None]  # [n, 1]
    is_left_j = is_left_j[None, :]  # [1, n]
    
    # 左臂 query (L_t):
    # - 可看 L_{<=t}: is_left_j & (t_j <= t_i)
    # - 可看 R_{<t}: ~is_left_j & (t_j < t_i)
    left_query_can_attend = (
        (is_left_j & (t_j <= t_i)) |  # L_t 看 L_{<=t}
        (~is_left_j & (t_j < t_i))    # L_t 看 R_{<t}
    )
    
    # 右臂 query (R_t):
    # - 可看 R_{<=t}: ~is_left_j & (t_j <= t_i)
    # - 可看 L_{<=t}: is_left_j & (t_j <= t_i)
    right_query_can_attend = (
        (~is_left_j & (t_j <= t_i)) |  # R_t 看 R_{<=t}
        (is_left_j & (t_j <= t_i))     # R_t 看 L_{<=t}
    )
    
    # 根据 query 是左臂还是右臂选择对应的 mask
    mask = jnp.where(is_left_i, left_query_can_attend, right_query_can_attend)
    
    return mask


def make_cross_vla_attn_mask(
    prefix_mask_left: jnp.ndarray,
    prefix_mask_right: jnp.ndarray,
    action_horizon: int,
) -> jnp.ndarray:
    """
    创建完整的cross-VLA attention mask，包括prefix和action tokens。
    
    完整的 token 布局:
    [left_prefix (pl)] [left_actions (ah)] [right_prefix (pr)] [right_actions (ah)]
    
    Attention 规则:
    1. 各自的 prefix 内部: 全双向 attention
    2. action tokens -> 自己的 prefix: 全部可见
    3. action tokens -> 对方的 prefix: 全部可见 (共享视觉/语言理解)
    4. action tokens 之间: 使用 dual_vla_ar_mask 的因果规则
    5. prefix tokens 不看 action tokens
    
    Args:
        prefix_mask_left: [batch, prefix_len_left] 左臂VLA的prefix有效mask
        prefix_mask_right: [batch, prefix_len_right] 右臂VLA的prefix有效mask
        action_horizon: 动作时间步数
        
    Returns:
        mask: [batch, total_len, total_len] attention mask
    """
    batch_size = prefix_mask_left.shape[0]
    pl = prefix_mask_left.shape[1]  # left prefix length
    pr = prefix_mask_right.shape[1]  # right prefix length
    ah = action_horizon
    total = pl + ah + pr + ah
    
    # 初始化 mask
    mask = jnp.zeros((batch_size, total, total), dtype=jnp.bool_)
    
    # === 1. Prefix 内部双向 attention ===
    # Left prefix 内部
    left_prefix_internal = prefix_mask_left[:, None, :] & prefix_mask_left[:, :, None]  # [b, pl, pl]
    mask = mask.at[:, :pl, :pl].set(left_prefix_internal)
    
    # Right prefix 内部
    right_prefix_start = pl + ah
    right_prefix_internal = prefix_mask_right[:, None, :] & prefix_mask_right[:, :, None]  # [b, pr, pr]
    mask = mask.at[:, right_prefix_start:right_prefix_start+pr, right_prefix_start:right_prefix_start+pr].set(right_prefix_internal)
    
    # === 2. Action tokens -> 自己的 prefix (全部可见) ===
    # Left actions -> Left prefix
    left_action_start = pl
    left_action_to_prefix = jnp.broadcast_to(
        prefix_mask_left[:, None, :],  # [b, 1, pl]
        (batch_size, ah, pl)
    )
    mask = mask.at[:, left_action_start:left_action_start+ah, :pl].set(left_action_to_prefix)
    
    # Right actions -> Right prefix
    right_action_start = pl + ah + pr
    right_action_to_prefix = jnp.broadcast_to(
        prefix_mask_right[:, None, :],  # [b, 1, pr]
        (batch_size, ah, pr)
    )
    mask = mask.at[:, right_action_start:right_action_start+ah, right_prefix_start:right_prefix_start+pr].set(right_action_to_prefix)
    
    # === 3. Action tokens -> 对方的 prefix (全部可见) ===
    # Left actions -> Right prefix
    mask = mask.at[:, left_action_start:left_action_start+ah, right_prefix_start:right_prefix_start+pr].set(
        jnp.broadcast_to(prefix_mask_right[:, None, :], (batch_size, ah, pr))
    )
    
    # Right actions -> Left prefix
    mask = mask.at[:, right_action_start:right_action_start+ah, :pl].set(
        jnp.broadcast_to(prefix_mask_left[:, None, :], (batch_size, ah, pl))
    )
    
    # === 4. Action tokens 之间的 causal joint attention ===
    action_ar_mask = make_dual_vla_ar_mask(ah)  # [2ah, 2ah]
    
    # 拆分成四个块:
    # - Left actions -> Left actions: action_ar_mask[:ah, :ah]
    # - Left actions -> Right actions: action_ar_mask[:ah, ah:]
    # - Right actions -> Left actions: action_ar_mask[ah:, :ah]
    # - Right actions -> Right actions: action_ar_mask[ah:, ah:]
    
    ll_mask = action_ar_mask[:ah, :ah]  # Left -> Left
    lr_mask = action_ar_mask[:ah, ah:]  # Left -> Right
    rl_mask = action_ar_mask[ah:, :ah]  # Right -> Left
    rr_mask = action_ar_mask[ah:, ah:]  # Right -> Right
    
    # 广播到 batch 维度并设置
    mask = mask.at[:, left_action_start:left_action_start+ah, left_action_start:left_action_start+ah].set(
        jnp.broadcast_to(ll_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, left_action_start:left_action_start+ah, right_action_start:right_action_start+ah].set(
        jnp.broadcast_to(lr_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, right_action_start:right_action_start+ah, left_action_start:left_action_start+ah].set(
        jnp.broadcast_to(rl_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, right_action_start:right_action_start+ah, right_action_start:right_action_start+ah].set(
        jnp.broadcast_to(rr_mask[None, :, :], (batch_size, ah, ah))
    )
    
    return mask


def make_suffix_cross_attn_mask(
    action_horizon: int,
    batch_size: int,
    left_suffix_len: int,
    right_suffix_len: int,
) -> jnp.ndarray:
    """
    创建简化版的suffix cross-attention mask。
    
    用于推理时，只关注 suffix tokens (state + actions) 之间的 attention。
    假设 left_suffix 和 right_suffix 长度相同 (都是 1 + action_horizon)。
    
    Token 布局:
    [left_state (1)] [left_actions (ah)] [right_state (1)] [right_actions (ah)]
    
    Args:
        action_horizon: 动作时间步数
        batch_size: batch 大小
        left_suffix_len: 左臂 suffix 长度 (通常是 1 + action_horizon)
        right_suffix_len: 右臂 suffix 长度
        
    Returns:
        mask: [batch, total_suffix, total_suffix] attention mask
    """
    ah = action_horizon
    total = left_suffix_len + right_suffix_len
    
    # 假设 suffix 格式是 [state_token, action_tokens...]
    # left: [0, 1:1+ah], right: [left_suffix_len, left_suffix_len+1:...]
    
    mask = jnp.zeros((batch_size, total, total), dtype=jnp.bool_)
    
    # State tokens 可以被所有 action tokens 看到
    # Left state (index 0)
    mask = mask.at[:, :, 0].set(True)
    # Right state (index left_suffix_len)
    mask = mask.at[:, :, left_suffix_len].set(True)
    
    # State tokens 只看自己
    mask = mask.at[:, 0, 0].set(True)
    mask = mask.at[:, left_suffix_len, left_suffix_len].set(True)
    
    # Action tokens 之间使用 causal joint attention
    # 左臂 actions: [1, 1+ah), 右臂 actions: [left_suffix_len+1, left_suffix_len+1+ah)
    action_ar_mask = make_dual_vla_ar_mask(ah)  # [2ah, 2ah]
    
    left_action_start = 1
    right_action_start = left_suffix_len + 1
    
    # 分解并设置
    ll_mask = action_ar_mask[:ah, :ah]
    lr_mask = action_ar_mask[:ah, ah:]
    rl_mask = action_ar_mask[ah:, :ah]
    rr_mask = action_ar_mask[ah:, ah:]
    
    mask = mask.at[:, left_action_start:left_action_start+ah, left_action_start:left_action_start+ah].set(
        jnp.broadcast_to(ll_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, left_action_start:left_action_start+ah, right_action_start:right_action_start+ah].set(
        jnp.broadcast_to(lr_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, right_action_start:right_action_start+ah, left_action_start:left_action_start+ah].set(
        jnp.broadcast_to(rl_mask[None, :, :], (batch_size, ah, ah))
    )
    mask = mask.at[:, right_action_start:right_action_start+ah, right_action_start:right_action_start+ah].set(
        jnp.broadcast_to(rr_mask[None, :, :], (batch_size, ah, ah))
    )
    
    # Actions 可以看 state tokens (已经在上面设置)
    # 额外确保 left actions 看 right state, right actions 看 left state
    mask = mask.at[:, left_action_start:left_action_start+ah, left_suffix_len].set(True)
    mask = mask.at[:, right_action_start:right_action_start+ah, 0].set(True)
    
    return mask


def visualize_mask(mask: jnp.ndarray, labels: list = None) -> str:
    """
    可视化 attention mask (用于调试)。
    
    Args:
        mask: [n, n] boolean mask
        labels: 可选的 token 标签列表
        
    Returns:
        字符串格式的可视化
    """
    n = mask.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    
    # 确保标签长度一致
    max_label_len = max(len(l) for l in labels)
    labels = [l.ljust(max_label_len) for l in labels]
    
    lines = []
    # Header
    header = " " * (max_label_len + 1) + " ".join([l[0] for l in labels])
    lines.append(header)
    
    # Body
    for i in range(n):
        row = labels[i] + " "
        for j in range(n):
            row += "█ " if mask[i, j] else "· "
        lines.append(row)
    
    return "\n".join(lines)
