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


class PaliGemmaWithDualExpertModel(nn.Module):
    """Dual-arm variant of PaliGemmaWithExpertModel with separate left/right action experts.
    
    This model splits the action expert into two independent experts for left and right arms,
    enabling cross-attention between them for coordinated bimanual control.
    
    Each arm can have its own LoRA configuration:
    - LoRA(L): Left arm action expert LoRA adapter
    - LoRA(R): Right arm action expert LoRA adapter
    
    The LoRA weights are named independently:
    - Left arm: gemma_expert_left.*.lora_a, gemma_expert_left.*.lora_b
    - Right arm: gemma_expert_right.*.lora_a, gemma_expert_right.*.lora_b
    """

    def __init__(
        self,
        vlm_config,
        action_expert_left_config,    # Config for left arm expert (may include LoRA(L))
        action_expert_right_config,   # Config for right arm expert (may include LoRA(R))
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False, False]  # [vlm, left_expert, right_expert]
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

        # Create config for left action expert with independent LoRA(L)
        action_expert_left_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_left_config.head_dim,
            hidden_size=action_expert_left_config.width,
            intermediate_size=action_expert_left_config.mlp_dim,
            num_attention_heads=action_expert_left_config.num_heads,
            num_hidden_layers=action_expert_left_config.depth,
            num_key_value_heads=action_expert_left_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_left_config.width if use_adarms[1] else None,
        )

        # Create config for right action expert with independent LoRA(R)
        action_expert_right_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_right_config.head_dim,
            hidden_size=action_expert_right_config.width,
            intermediate_size=action_expert_right_config.mlp_dim,
            num_attention_heads=action_expert_right_config.num_heads,
            num_hidden_layers=action_expert_right_config.depth,
            num_key_value_heads=action_expert_right_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[2],
            adarms_cond_dim=action_expert_right_config.width if use_adarms[2] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        
        # Two separate action experts for left and right arms
        # Each expert has its own LoRA adapter (LoRA(L) and LoRA(R))
        self.gemma_expert_left = GemmaForCausalLM(config=action_expert_left_config_hf)
        self.gemma_expert_right = GemmaForCausalLM(config=action_expert_right_config_hf)
        
        # Remove embedding tokens as we use custom embeddings
        self.gemma_expert_left.model.embed_tokens = None
        self.gemma_expert_right.model.embed_tokens = None

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
        alpha_t: torch.Tensor | None = None,
    ):
        """Forward pass for dual-arm model with gated cross-attention.
        
        Args:
            inputs_embeds: List of [prefix_embs, left_arm_embs, right_arm_embs]
            adarms_cond: List of [vlm_cond, left_expert_cond, right_expert_cond]
            alpha_t: Gate value [B, 1] controlling cross-arm attention:
                - alpha_t = 0 (or None): Full cross-attention between arms
                - alpha_t = 1: Independent attention (no cross-arm communication)
                - 0 < alpha_t < 1: Soft interpolation (training mode)
        """
        if adarms_cond is None:
            adarms_cond = [None, None, None]
        
        # Case 1: Only prefix (VLM) forward - for caching
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
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
            return [prefix_output, None, None], prefix_past_key_values
        
        # Case 2: Only suffix (experts) forward - using cached prefix
        elif inputs_embeds[0] is None:
            # Process left and right experts independently with cross-attention via shared KV
            left_output = self.gemma_expert_left.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            right_output = self.gemma_expert_right.model.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[2] if adarms_cond is not None else None,
            )
            return [None, left_output.last_hidden_state, right_output.last_hidden_state], None
        
        # Case 3: Full forward with all three branches and gated cross-attention
        else:
            models = [
                self.paligemma.language_model,
                self.gemma_expert_left.model,
                self.gemma_expert_right.model,
            ]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert_left.model, "gradient_checkpointing")
                and self.gemma_expert_left.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if in training mode
            if self.training:
                for expert in [self.gemma_expert_left.model, self.gemma_expert_right.model]:
                    if hasattr(expert, "gradient_checkpointing") and not expert.gradient_checkpointing:
                        expert.gradient_checkpointing = True
                        use_gradient_checkpointing = True

            def compute_layer_cross_attention(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                """Compute one layer with full cross-attention between all three branches."""
                models = [
                    self.paligemma.language_model,
                    self.gemma_expert_left.model,
                    self.gemma_expert_right.model,
                ]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                normalized_hidden_states = []
                
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    norm_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                    gates.append(gate)
                    normalized_hidden_states.append(norm_hidden)

                    input_shape = norm_hidden.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(norm_hidden).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(norm_hidden).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(norm_hidden).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate all three branches for cross-attention
                query_states_cat = torch.cat(query_states, dim=2)
                key_states_cat = torch.cat(key_states, dim=2)
                value_states_cat = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states_cat.shape[0],
                    query_states_cat.shape[2],
                    query_states_cat.shape[-1],
                    device=query_states_cat.device,
                    dtype=query_states_cat.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states_cat, key_states_cat = modeling_gemma.apply_rotary_pos_emb(
                    query_states_cat, key_states_cat, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states_cat.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Cross-attention computation across all three branches
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states_cat,
                    key_states_cat,
                    value_states_cat,
                    attention_mask,
                    scaling,
                )
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process outputs for each branch
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # First residual connection
                    out_emb = modeling_gemma._gated_residual(normalized_hidden_states[i], out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # Second residual connection
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            def compute_layer_independent(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                """Compute one layer with independent attention (no cross-arm communication).
                
                VLM still sees both arms, but left/right arms only see themselves + VLM.
                """
                models = [
                    self.paligemma.language_model,
                    self.gemma_expert_left.model,
                    self.gemma_expert_right.model,
                ]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                normalized_hidden_states = []
                
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    norm_hidden, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
                    gates.append(gate)
                    normalized_hidden_states.append(norm_hidden)

                    input_shape = norm_hidden.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(norm_hidden).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(norm_hidden).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(norm_hidden).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Get sequence lengths
                prefix_len = inputs_embeds[0].shape[1]
                left_len = inputs_embeds[1].shape[1]
                right_len = inputs_embeds[2].shape[1]
                batch_size = inputs_embeds[0].shape[0]
                
                # Apply rotary embeddings
                dummy_tensor = torch.zeros(
                    batch_size,
                    prefix_len + left_len + right_len,
                    query_states[0].shape[-1],
                    device=query_states[0].device,
                    dtype=query_states[0].dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                
                # Apply rotary to each branch separately
                for i in range(3):
                    if i == 0:
                        pos_slice = slice(0, prefix_len)
                    elif i == 1:
                        pos_slice = slice(prefix_len, prefix_len + left_len)
                    else:
                        pos_slice = slice(prefix_len + left_len, prefix_len + left_len + right_len)
                    
                    cos_i = cos[:, pos_slice, :]
                    sin_i = sin[:, pos_slice, :]
                    query_states[i], key_states[i] = modeling_gemma.apply_rotary_pos_emb(
                        query_states[i], key_states[i], cos_i, sin_i, unsqueeze_dim=1
                    )

                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                
                outputs_embeds = []
                
                # VLM: attends to all (prefix + left + right)
                q_vlm = query_states[0]
                k_vlm = torch.cat([key_states[0], key_states[1], key_states[2]], dim=2)
                v_vlm = torch.cat([value_states[0], value_states[1], value_states[2]], dim=2)
                
                # Create attention mask for VLM (can see everything)
                vlm_attn_mask = attention_mask[:, :, :prefix_len, :] if attention_mask is not None else None
                
                att_vlm, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    q_vlm, k_vlm, v_vlm, vlm_attn_mask, scaling,
                )
                att_vlm = att_vlm.reshape(batch_size, prefix_len, 1 * 8 * head_dim)
                
                # Left arm: attends to prefix + left only (not right)
                q_left = query_states[1]
                k_left = torch.cat([key_states[0], key_states[1]], dim=2)
                v_left = torch.cat([value_states[0], value_states[1]], dim=2)
                
                left_attn_mask = attention_mask[:, :, prefix_len:prefix_len+left_len, :prefix_len+left_len] if attention_mask is not None else None
                
                att_left, _ = modeling_gemma.eager_attention_forward(
                    self.gemma_expert_left.model.layers[layer_idx].self_attn,
                    q_left, k_left, v_left, left_attn_mask, scaling,
                )
                att_left = att_left.reshape(batch_size, left_len, 1 * 8 * head_dim)
                
                # Right arm: attends to prefix + right only (not left)
                q_right = query_states[2]
                k_right = torch.cat([key_states[0], key_states[2]], dim=2)
                v_right = torch.cat([value_states[0], value_states[2]], dim=2)
                
                # For right arm, we need to adjust the mask indices
                if attention_mask is not None:
                    # Right arm mask: rows for right tokens, columns for prefix + right
                    right_attn_mask = torch.cat([
                        attention_mask[:, :, prefix_len+left_len:, :prefix_len],  # prefix part
                        attention_mask[:, :, prefix_len+left_len:, prefix_len+left_len:]  # right part
                    ], dim=-1)
                else:
                    right_attn_mask = None
                
                att_right, _ = modeling_gemma.eager_attention_forward(
                    self.gemma_expert_right.model.layers[layer_idx].self_attn,
                    q_right, k_right, v_right, right_attn_mask, scaling,
                )
                att_right = att_right.reshape(batch_size, right_len, 1 * 8 * head_dim)
                
                att_outputs = [att_vlm, att_left, att_right]
                
                # Process outputs for each branch
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    att_out = att_outputs[i]

                    if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_out)

                    # First residual connection
                    out_emb = modeling_gemma._gated_residual(normalized_hidden_states[i], out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # Second residual connection
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)

                return outputs_embeds

            def compute_layer_gated(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, alpha_t):
                """Compute one layer with gated cross-attention.
                
                Interpolates between cross-attention (alpha_t=0) and independent (alpha_t=1).
                """
                # Get outputs from both modes
                cross_outputs = compute_layer_cross_attention(
                    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                )
                indep_outputs = compute_layer_independent(
                    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                )
                
                # Soft interpolation: output = (1 - alpha_t) * cross + alpha_t * independent
                # alpha_t: [B, 1] -> need to broadcast
                outputs_embeds = []
                for i in range(3):
                    # Expand alpha_t for broadcasting: [B, 1] -> [B, 1, 1]
                    alpha_expanded = alpha_t.unsqueeze(-1) if alpha_t.dim() == 2 else alpha_t.unsqueeze(-1).unsqueeze(-1)
                    interpolated = (1 - alpha_expanded) * cross_outputs[i] + alpha_expanded * indep_outputs[i]
                    outputs_embeds.append(interpolated)
                
                return outputs_embeds

            # Process all layers
            for layer_idx in range(num_layers):
                # Determine which computation to use based on alpha_t
                if alpha_t is None:
                    # No gate, use full cross-attention
                    compute_fn = compute_layer_cross_attention
                    compute_args = (layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)
                elif torch.all(alpha_t == 0):
                    # Gate fully open, use cross-attention
                    compute_fn = compute_layer_cross_attention
                    compute_args = (layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)
                elif torch.all(alpha_t == 1):
                    # Gate fully closed, use independent
                    compute_fn = compute_layer_independent
                    compute_args = (layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond)
                else:
                    # Soft gate, use interpolation
                    compute_fn = compute_layer_gated
                    compute_args = (layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, alpha_t)
                
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_fn,
                        *compute_args,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_fn(*compute_args)

            # Final layer normalization
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            left_output = outputs_embeds[1]
            right_output = outputs_embeds[2]

        return [prefix_output, left_output, right_output], None
