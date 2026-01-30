from openai import OpenAI
import base64
import sys

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --------- YOUR INPUT PROMPT HERE (CLI) ---------
if len(sys.argv) < 2:
    raise SystemExit('Usage: python run_qwen_vl.py "<TASK_PROMPT>"')
task_prompt = sys.argv[1]
# -----------------------------------------------

# Image order matters: 1) CENTER, 2) LEFT, 3) RIGHT
images = [
    img_to_b64("/vepfs-mlp2/mlp-public/haoce/zxr/mid2.jpg"),
    img_to_b64("/vepfs-mlp2/mlp-public/haoce/zxr/left2.jpg"),
    img_to_b64("/vepfs-mlp2/mlp-public/haoce/zxr/right2.jpg"),
]

user_text = f"""
You are given THREE images of the same task scene:

- Image 1: CENTER view (global scene understanding, object layout, overall constraints)
- Image 2: LEFT-ARM view (what the left arm sees; use this to decide left arm grasp points, reachability, and collisions)
- Image 3: RIGHT-ARM view (what the right arm sees; use this to decide right arm grasp points, reachability, and collisions)

You MUST use the correct view information:
- Use CENTER view for global planning and ensuring both arms do not collide.
- Use LEFT-ARM view primarily to generate the left_arm_prompt.
- Use RIGHT-ARM view primarily to generate the right_arm_prompt.

Your goals:
1) Produce a left_arm_prompt. Keep it short and accurate.
2) Produce a right_arm_prompt. Keep it short and accurate.
3) Produce a cooperation_score between 0 and 1.

Cooperation score definition:
- 0.0–0.2: Pure single-arm task (other arm only idle or avoids collision)
- 0.2–0.4: Single-arm dominant, other arm optional for stability
- 0.4–0.6: Weak cooperation (one arm fixes, the other operates)
- 0.6–0.8: Strong cooperation (synchronized or dual-point manipulation)
- 0.8–1.0: Highly coupled bimanual manipulation (both arms essential)

Output STRICT JSON ONLY (no extra text):

{{
  "left_arm_prompt": "...",
  "right_arm_prompt": "...",
  "cooperation_score": 0.0,
  "cooperation_reason": "..."
}}

Task description:
<<<
{task_prompt}
>>>
""".strip()

messages = [
    {"role": "system", "content": "You are a dual-arm robot task decomposition and coordination evaluator."},
    {
        "role": "user",
        "content": (
            [{"type": "text", "text": user_text}]
            + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images]
        ),
    },
]

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages,
    temperature=0.2,
    max_tokens=512,  # 如报参数名错误，改 max_completion_tokens=512
)

print(resp.choices[0].message.content)
