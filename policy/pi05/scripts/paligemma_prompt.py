import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.models.pi0 import make_attn_mask
from openpi.shared import image_tools

PROMPT = "Describe the object in the image."
CHECKPOINT_PATH = None  # e.g. pathlib.Path("/path/to/params")
PALIGEMMA_VARIANT = "gemma_2b"
ACTION_EXPERT_VARIANT = "gemma_300m"
MAX_NEW_TOKENS = 16
IMAGE_PATHS = {
    "base_0_rgb": pathlib.Path("/path/to/base_0_rgb.png"),
    "left_wrist_0_rgb": pathlib.Path("/path/to/left_wrist_0_rgb.png"),
    "right_wrist_0_rgb": pathlib.Path("/path/to/right_wrist_0_rgb.png"),
}


def _load_image(image_path: pathlib.Path) -> jax.Array:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    image = image / 255.0 * 2.0 - 1.0
    image = jnp.asarray(image)
    return image_tools.resize_with_pad(image, _model.IMAGE_RESOLUTION[0], _model.IMAGE_RESOLUTION[1])


def _build_observation(prompt: str, image_paths: dict[str, pathlib.Path]) -> _model.Observation:
    tokenizer = _tokenizer.PaligemmaTokenizer()
    tokens, token_mask = tokenizer.tokenize(prompt)
    tokenized_prompt = jnp.asarray(tokens)[None, :]
    tokenized_prompt_mask = jnp.asarray(token_mask)[None, :]

    images = {key: _load_image(path)[None, ...] for key, path in image_paths.items()}
    image_masks = {key: jnp.ones((1,), dtype=jnp.bool_) for key in image_paths}
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=jnp.zeros((1, 32), dtype=jnp.float32),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(image_paths.keys()))
    return observation


def _greedy_decode(model, observation: _model.Observation, max_new_tokens: int) -> str:
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=model.max_token_len)
    prompt_tokens = observation.tokenized_prompt
    prompt_mask = observation.tokenized_prompt_mask
    prompt_len = int(prompt_mask.sum())
    token_ids = prompt_tokens[:, :prompt_len]

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    generated_tokens = []

    for _ in range(max_new_tokens):
        token_embeddings = model.PaliGemma.llm(token_ids, method="embed")
        full_tokens = jnp.concatenate([prefix_tokens, token_embeddings], axis=1)
        token_mask_full = jnp.concatenate(
            [
                prompt_mask[:, :prompt_len],
                jnp.ones((1, token_ids.shape[1] - prompt_len), dtype=jnp.bool_),
            ],
            axis=1,
        )
        full_mask = jnp.concatenate([prefix_mask, token_mask_full], axis=1)
        prompt_ar = jnp.zeros((prompt_len,), dtype=jnp.bool_)
        generated_ar = jnp.ones((token_ids.shape[1] - prompt_len,), dtype=jnp.bool_)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, prompt_ar, generated_ar], axis=0)
        attn_mask = make_attn_mask(full_mask, full_ar_mask)
        positions = jnp.cumsum(full_mask, axis=1) - 1
        (hidden, _), _ = model.PaliGemma.llm(
            [full_tokens, None], mask=attn_mask, positions=positions, adarms_cond=[None, None]
        )
        logits = model.PaliGemma.llm(hidden, method="decode")
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_tokens.append(next_token)
        token_ids = jnp.concatenate([token_ids, next_token[:, None]], axis=1)

    generated_ids = jnp.concatenate(generated_tokens, axis=0)[None, :]
    return tokenizer._tokenizer.decode(generated_ids[0].tolist())


def main() -> None:
    config = pi0_config.Pi0Config(
        paligemma_variant=PALIGEMMA_VARIANT,
        action_expert_variant=ACTION_EXPERT_VARIANT,
    )
    model = config.create(jax.random.key(0))
    if CHECKPOINT_PATH is not None:
        params = _model.restore_params(CHECKPOINT_PATH, dtype=jnp.bfloat16)
        model = config.load(params)

    observation = _build_observation(PROMPT, IMAGE_PATHS)
    output_text = _greedy_decode(model, observation, MAX_NEW_TOKENS)
    print(output_text)


if __name__ == "__main__":
    main()
