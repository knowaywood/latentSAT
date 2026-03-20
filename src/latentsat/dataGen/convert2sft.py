"""
步骤 1：将 cnf_cot_dataset.jsonl 转换为 Qwen3 SFT 格式
输出格式：每行一个 JSON，包含 messages 列表（system / user / assistant）
"""

import json
import random

SYSTEM_PROMPT = (
    "你是一个逻辑推理专家，擅长分析合取范式（CNF）的可满足性问题（SAT）。"
    "请先逐步展示完整的推理过程，最后给出明确结论。"
)

# 多样化的 instruction 模板，避免模型过拟合单一提问方式
INSTRUCTION_TEMPLATES = [
    "请判断以下 CNF 公式是否可满足，并给出详细的推理步骤：\n\n{formula}",
    "以下是一个合取范式（CNF）公式，请用 DPLL 方法求解，并逐步说明推理过程：\n\n{formula}",
    "判断该命题逻辑公式的可满足性，需要展示完整的求解过程：\n\n{formula}",
    "对以下 CNF 公式进行 SAT 求解，请一步一步地分析：\n\n{formula}",
]


def extract_formula(text: str) -> str:
    """从 text 字段中提取公式部分"""
    for line in text.split("\n"):
        if line.startswith("给定合取范式"):
            # 取冒号后面的公式
            if "：" in line:
                return line.split("：", 1)[1].strip()
    return ""


def extract_reasoning_and_conclusion(text: str) -> str:
    """提取【求解过程】到结尾的部分作为 assistant 回复"""
    if "【求解过程】" in text:
        return text[text.index("【求解过程】") :].strip()
    return text.strip()


def convert(input_path: str, output_path: str, seed: int = 42):
    random.seed(seed)
    samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            text = raw["text"]
            meta = raw["meta"]

            formula = extract_formula(text)
            if not formula:
                continue

            # 随机选一个 instruction 模板
            template = random.choice(INSTRUCTION_TEMPLATES)
            user_msg = template.format(formula=formula)

            # assistant 回复 = 推理过程 + 结论
            assistant_msg = extract_reasoning_and_conclusion(text)

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "meta": meta,  # 保留元数据，训练时可忽略
            }
            samples.append(sample)

    # 打乱顺序
    random.shuffle(samples)

    # 按 9:1 切分 train / eval
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    eval_samples = samples[split:]

    # 写入文件
    train_path = output_path.replace(".jsonl", "_train.jsonl")
    eval_path = output_path.replace(".jsonl", "_eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("✅ 转换完成")
    print(f"   训练集：{train_path}  ({len(train_samples)} 条)")
    print(f"   验证集：{eval_path}  ({len(eval_samples)} 条)")

    # 打印一条示例
    print("\n=== 示例样本 ===")
    s = train_samples[0]
    for msg in s["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:300] + ("..." if len(msg["content"]) > 300 else "")
        print(f"\n[{role}]\n{content}")


if __name__ == "__main__":
    convert(
        input_path="./data/cnf_cot_dataset.jsonl",
        output_path="data/cnf_sft.jsonl",
    )
