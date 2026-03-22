from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from latentsat.model import LatentSATConfig, LatentSATModel, build_model
from latentsat.utils.verify import VerifyList

# 初始化验证器
verify = VerifyList()


def load_pretrained_model(
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> LatentSATModel:
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / "config.json"
    model_path = checkpoint_path / "model.pt"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Checkpoint files missing in {checkpoint_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    model = build_model(LatentSATConfig(**config_dict))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation for LatentSAT")
    parser.add_argument("--test-file", type=str, default="./data/list_data_eval.jsonl")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        # default="checkpoints/list_struct/pretrain/epoch_10",
        default="checkpoints/list_struct/rl/epoch_8",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_jsonl_data(file_path: str | Path) -> list[dict]:
    """流式加载 jsonl 数据，避免一次性读取大文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data[:4000]


def main() -> None:
    args = parse_args()

    # 1. 加载数据
    print(f"Loading data from {args.test_file}...")
    dataset = load_jsonl_data(args.test_file)

    # 2. 加载模型
    print(f"Loading model from {args.pretrained_dir}...")
    model = load_pretrained_model(args.pretrained_dir, device=args.device)

    # 3. 评估循环
    correct_count = 0
    total_count = len(dataset)

    # 使用 tqdm 并添加后缀显示实时准确率
    pbar = tqdm(dataset, desc="Evaluating", unit="samples")

    with torch.no_grad():  # 禁用梯度计算，加速推理
        for item in pbar:
            # 执行推理
            outputs = model(item["clauses"])

            # 验证结果
            is_correct = verify.verify(
                item["clauses"],
                item["num_vars"],
                outputs.structured_output,
                item["satisfiable"],
            )

            if is_correct:
                correct_count += 1

            # 动态更新进度条右侧信息
            pbar.set_postfix(acc=f"{correct_count / (pbar.n + 1):.2%}")

    # 4. 输出最终结果
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "=" * 30)
    print(f"Final Accuracy: {final_accuracy:.4f} ({correct_count}/{total_count})")
    print("=" * 30)


if __name__ == "__main__":
    main()
