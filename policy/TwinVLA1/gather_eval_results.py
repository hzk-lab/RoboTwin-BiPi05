"""
汇总 TwinVLA1 评测结果：
- 收集 eval_result 下的 mp4 到一个汇总目录 (硬链接，省空间)
- 尝试读取 success 标记，统计成功率

默认根目录：/data0/users/haoce/RoboTwin/eval_result
默认输出目录：/data0/users/haoce/RoboTwin/eval_result/_merged

成功标记优先级：
1) episode*/result.json 里字段 success 或 success_rate
2) episode*/success.json / success.txt (包含 "success" 或 "True")
3) 如果找不到，计为未知
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT_DEFAULT = Path("/data0/users/haoce/RoboTwin/eval_result")
OUTPUT_DEFAULT = ROOT_DEFAULT / "_merged"


def read_success(ep_dir: Path):
    # 尝试多种文件
    candidates = ["result.json", "success.json", "success.txt", "info.json"]
    for name in candidates:
        f = ep_dir / name
        if not f.exists():
            continue
        try:
            if f.suffix == ".json":
                data = json.loads(f.read_text())
                # 常见字段
                for k in ["success", "success_rate", "succeed", "is_success"]:
                    if k in data:
                        val = data[k]
                        if isinstance(val, (bool, int, float)):
                            return bool(val), k
                # 如果是数字，>0.5 当成功
                for v in data.values():
                    if isinstance(v, (int, float)):
                        return (v > 0.5), "numeric"
            else:
                txt = f.read_text().lower()
                if "success" in txt or "true" in txt:
                    return True, "text"
                if "fail" in txt or "false" in txt:
                    return False, "text"
        except Exception:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="eval_result 根目录")
    ap.add_argument("--out", type=Path, default=OUTPUT_DEFAULT, help="汇总输出目录")
    args = ap.parse_args()

    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    episodes = sorted(root.glob("**/episode*"))
    total = 0
    success = 0
    unknown = 0
    records = []

    for ep in episodes:
        # 找 mp4
        videos = sorted(ep.glob("*.mp4"))
        if not videos:
            continue
        total += 1
        succ, src = read_success(ep)
        if succ is True:
            success += 1
        elif succ is None:
            unknown += 1
        # 取第一个视频做汇总
        v = videos[0]
        rel = ep.relative_to(root)
        dest_dir = out / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / v.name
        if dest.exists():
            dest.unlink()
        # 硬链接节省空间，失败则拷贝
        try:
            os.link(v, dest)
        except OSError:
            shutil.copy2(v, dest)
        records.append({"episode": str(rel), "video": str(dest), "success": succ, "source": src})

    summary = {
        "total_episodes": total,
        "success": success,
        "unknown": unknown,
        "fail": total - success - unknown,
        "success_rate_known": f"{(success/(total-unknown)*100):.2f}%" if total>unknown else "n/a",
    }
    (out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": records}, indent=2))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
