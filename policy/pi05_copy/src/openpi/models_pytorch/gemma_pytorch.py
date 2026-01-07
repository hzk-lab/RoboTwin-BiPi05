from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class DualActionExpert(nn.Module):
    """Dual Action Expert module for left and right arms with cross-attention.
    
    Each AE attends to its respective Per-Arm VLM output, with optional
    cross-attention between the two AEs scaled by alpha_t.
    """
    
    def __init__(
        self,
        ae_left_config,
        ae_right_config,
        use_adarms: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize the Dual Action Expert.
        
        Args:
            ae_left_config: Configuration for left arm AE.
            ae_right_config: Configuration for right arm AE.
            use_adarms: Whether to use adaRMSNorm for timestep injection.
            precision: Model precision.
        """
        super().__init__()
        
        # Left AE config
        ae_left_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=ae_left_config.head_dim,
            hidden_size=ae_left_config.width,
            intermediate_size=ae_left_config.mlp_dim,
            num_attention_heads=ae_left_config.num_heads,
            num_hidden_layers=ae_left_config.depth,
            num_key_value_heads=ae_left_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms,
            adarms_cond_dim=ae_left_config.width if use_adarms else None,
        )
        
        # Right AE config
        ae_right_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=ae_right_config.head_dim,
            hidden_size=ae_right_config.width,
            intermediate_size=ae_right_config.mlp_dim,
            num_attention_heads=ae_right_config.num_heads,
            num_hidden_layers=ae_right_config.depth,
            num_key_value_heads=ae_right_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms,
            adarms_cond_dim=ae_right_config.width if use_adarms else None,
        )
        
        # Initialize AEs
        self.ae_left = GemmaForCausalLM(config=ae_left_config_hf)
        self.ae_right = GemmaForCausalLM(config=ae_right_config_hf)
        
        # Remove embedding layers (we use external embeddings)
        self.ae_left.model.embed_tokens = None
        self.ae_right.model.embed_tokens = None
        
        # Store dimensions
        self.hidden_dim = ae_left_config.width
        self.num_layers = ae_left_config.depth
        self.num_heads = ae_left_config.num_heads
        self.head_dim = ae_left_config.head_dim
        
        # Apply precision
        self._apply_precision(precision)
    
    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        """Apply the specified precision to the model."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            params_to_keep_float32 = [
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
        elif precision == "float32":
            self.to(dtype=torch.float32)
    
    def forward_with_cross_attention(
        self,
        vlm_left_out: torch.Tensor,
        vlm_right_out: torch.Tensor,
        suffix_left_embs: torch.Tensor,
        suffix_right_embs: torch.Tensor,
        attention_mask_left: torch.Tensor,
        attention_mask_right: torch.Tensor,
        position_ids: torch.Tensor,
        alpha_t: torch.Tensor | None = None,
        adarms_cond_left: torch.Tensor | None = None,
        adarms_cond_right: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional cross-attention between arms.
        
        Args:
            vlm_left_out: Left Per-Arm VLM output [batch, vlm_len, hidden_dim]
            vlm_right_out: Right Per-Arm VLM output [batch, vlm_len, hidden_dim]
            suffix_left_embs: Left arm action embeddings [batch, action_len, hidden_dim]
            suffix_right_embs: Right arm action embeddings [batch, action_len, hidden_dim]
            attention_mask_left: Attention mask for left arm [batch, 1, total_len, total_len]
            attention_mask_right: Attention mask for right arm [batch, 1, total_len, total_len]
            position_ids: Position IDs [batch, total_len]
            alpha_t: Gate value [batch, 1], 0=full cooperation, 1=independent
            adarms_cond_left: adaRMS conditioning for left arm [batch, hidden_dim]
            adarms_cond_right: adaRMS conditioning for right arm [batch, hidden_dim]
        
        Returns:
            Tuple of (left_output, right_output), each [batch, action_len, hidden_dim]
        """
        # Concatenate VLM output with action embeddings for each arm
        # prefix = VLM output, suffix = action embeddings
        inputs_left = torch.cat([vlm_left_out, suffix_left_embs], dim=1)
        inputs_right = torch.cat([vlm_right_out, suffix_right_embs], dim=1)
        
        # If alpha_t is None, default to independent (no cross-attention)
        if alpha_t is None:
            alpha_t = torch.ones(inputs_left.shape[0], 1, device=inputs_left.device)
        
        # Process through layers
        hidden_left = inputs_left
        hidden_right = inputs_right
        
        for layer_idx in range(self.num_layers):
            hidden_left, hidden_right = self._forward_layer_with_cross_attn(
                layer_idx,
                hidden_left,
                hidden_right,
                attention_mask_left,
                attention_mask_right,
                position_ids,
                alpha_t,
                adarms_cond_left,
                adarms_cond_right,
            )
        
        # Final norm
        hidden_left, _ = self.ae_left.model.norm(hidden_left, cond=adarms_cond_left)
        hidden_right, _ = self.ae_right.model.norm(hidden_right, cond=adarms_cond_right)
        
        # Extract only action outputs (suffix portion)
        vlm_len = vlm_left_out.shape[1]
        output_left = hidden_left[:, vlm_len:]
        output_right = hidden_right[:, vlm_len:]
        
        return output_left, output_right
    
    def _forward_layer_with_cross_attn(
        self,
        layer_idx: int,
        hidden_left: torch.Tensor,
        hidden_right: torch.Tensor,
        attention_mask_left: torch.Tensor,
        attention_mask_right: torch.Tensor,
        position_ids: torch.Tensor,
        alpha_t: torch.Tensor,
        adarms_cond_left: torch.Tensor | None,
        adarms_cond_right: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one layer with cross-attention between arms."""
        layer_left = self.ae_left.model.layers[layer_idx]
        layer_right = self.ae_right.model.layers[layer_idx]
        
        # Input layer norm
        normed_left, gate_left = layer_left.input_layernorm(hidden_left, cond=adarms_cond_left)
        normed_right, gate_right = layer_right.input_layernorm(hidden_right, cond=adarms_cond_right)
        
        # Compute Q, K, V for both arms
        batch_size = hidden_left.shape[0]
        seq_len = hidden_left.shape[1]
        
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        
        q_left = layer_left.self_attn.q_proj(normed_left).view(hidden_shape).transpose(1, 2)
        k_left = layer_left.self_attn.k_proj(normed_left).view(hidden_shape).transpose(1, 2)
        v_left = layer_left.self_attn.v_proj(normed_left).view(hidden_shape).transpose(1, 2)
        
        q_right = layer_right.self_attn.q_proj(normed_right).view(hidden_shape).transpose(1, 2)
        k_right = layer_right.self_attn.k_proj(normed_right).view(hidden_shape).transpose(1, 2)
        v_right = layer_right.self_attn.v_proj(normed_right).view(hidden_shape).transpose(1, 2)
        
        # Apply rotary embeddings
        dummy = torch.zeros(batch_size, seq_len, self.head_dim, device=hidden_left.device, dtype=hidden_left.dtype)
        cos, sin = self.ae_left.model.rotary_emb(dummy, position_ids)
        
        q_left, k_left = modeling_gemma.apply_rotary_pos_emb(q_left, k_left, cos, sin, unsqueeze_dim=1)
        q_right, k_right = modeling_gemma.apply_rotary_pos_emb(q_right, k_right, cos, sin, unsqueeze_dim=1)
        
        # Self-attention (independent)
        scaling_left = layer_left.self_attn.scaling
        scaling_right = layer_right.self_attn.scaling
        
        attn_out_left_self, _ = modeling_gemma.eager_attention_forward(
            layer_left.self_attn, q_left, k_left, v_left, attention_mask_left, scaling_left
        )
        attn_out_right_self, _ = modeling_gemma.eager_attention_forward(
            layer_right.self_attn, q_right, k_right, v_right, attention_mask_right, scaling_right
        )
        
        # Cross-attention (when alpha_t < 1)
        # Left queries attend to right keys/values
        attn_out_left_cross, _ = modeling_gemma.eager_attention_forward(
            layer_left.self_attn, q_left, k_right, v_right, attention_mask_right, scaling_left
        )
        # Right queries attend to left keys/values
        attn_out_right_cross, _ = modeling_gemma.eager_attention_forward(
            layer_right.self_attn, q_right, k_left, v_left, attention_mask_left, scaling_right
        )
        
        # Interpolate between self and cross attention based on alpha_t
        # alpha_t = 1: fully independent (only self attention)
        # alpha_t = 0: full cooperation (mix of self and cross attention)
        alpha_t_expanded = alpha_t.view(batch_size, 1, 1, 1)
        
        # For simplicity, we blend self and cross outputs
        # When alpha=1: use only self attention
        # When alpha=0: use average of self and cross attention
        attn_out_left = alpha_t_expanded * attn_out_left_self + (1 - alpha_t_expanded) * (
            0.5 * attn_out_left_self + 0.5 * attn_out_left_cross
        )
        attn_out_right = alpha_t_expanded * attn_out_right_self + (1 - alpha_t_expanded) * (
            0.5 * attn_out_right_self + 0.5 * attn_out_right_cross
        )
        
        # Reshape attention outputs
        attn_out_left = attn_out_left.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out_right = attn_out_right.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        # Output projection
        if attn_out_left.dtype != layer_left.self_attn.o_proj.weight.dtype:
            attn_out_left = attn_out_left.to(layer_left.self_attn.o_proj.weight.dtype)
            attn_out_right = attn_out_right.to(layer_right.self_attn.o_proj.weight.dtype)
        
        attn_out_left = layer_left.self_attn.o_proj(attn_out_left)
        attn_out_right = layer_right.self_attn.o_proj(attn_out_right)
        
        # First residual
        hidden_left = modeling_gemma._gated_residual(normed_left, attn_out_left, gate_left)
        hidden_right = modeling_gemma._gated_residual(normed_right, attn_out_right, gate_right)
        
        after_first_residual_left = hidden_left.clone()
        after_first_residual_right = hidden_right.clone()
        
        # Post-attention layer norm
        normed_left, gate_left = layer_left.post_attention_layernorm(hidden_left, cond=adarms_cond_left)
        normed_right, gate_right = layer_right.post_attention_layernorm(hidden_right, cond=adarms_cond_right)
        
        # Convert to bfloat16 if needed
        if layer_left.mlp.up_proj.weight.dtype == torch.bfloat16:
            normed_left = normed_left.to(dtype=torch.bfloat16)
            normed_right = normed_right.to(dtype=torch.bfloat16)
        
        # MLP
        mlp_out_left = layer_left.mlp(normed_left)
        mlp_out_right = layer_right.mlp(normed_right)
        
        # Second residual
        hidden_left = modeling_gemma._gated_residual(after_first_residual_left, mlp_out_left, gate_left)
        hidden_right = modeling_gemma._gated_residual(after_first_residual_right, mlp_out_right, gate_right)
        
        return hidden_left, hidden_right
