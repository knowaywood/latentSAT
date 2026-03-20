"""
CNF CoT 数据集生成器
用途：LLM 预训练 / JSONL 格式
每条样本包含：CNF 公式描述 + 思维链推理过程 + 最终答案
"""

import json
import os
import random
from typing import Optional

from latentsat.utils.cdcl import cdcl

# ─────────────────────────────────────────────
# 1. CNF 随机生成
# ─────────────────────────────────────────────


def generate_cnf(
    num_vars: int, num_clauses: int, clause_size: int = 3
) -> list[list[int]]:
    """随机生成一个 CNF 公式（子句列表，每个子句是文字列表）"""
    clauses = []
    for _ in range(num_clauses):
        vars_chosen = random.sample(range(1, num_vars + 1), min(clause_size, num_vars))
        clause = [v if random.random() > 0.5 else -v for v in vars_chosen]
        clauses.append(clause)
    return clauses


def generate_unsat_cnf(num_vars: int, clause_size: int = 3) -> list[list[int]]:
    """
    强制生成一个 UNSAT 的 CNF 公式。
    策略：先随机选一个完整赋值，对其取反后添加所有组合子句，
    并额外添加高密度随机子句使其过约束。
    """
    # 使用高子句密度（远超相变阈值）强制生成 UNSAT
    ratio = random.uniform(6.0, 9.0)  # 3-SAT 相变点 ~4.27，远超则大概率 UNSAT
    num_clauses = int(num_vars * ratio)
    for _ in range(200):  # 多次尝试
        clauses = generate_cnf(num_vars, num_clauses, clause_size)
        is_sat, _, _ = solve_cnf(clauses, num_vars)
        if not is_sat:
            return clauses
    # 保底：直接构造矛盾（x1 ∧ ¬x1）
    base = generate_cnf(num_vars, num_clauses // 2, clause_size)
    base.append([1])  # x1 = True
    base.append([-1])  # x1 = False
    return base


def solve_cnf(
    clauses: list[list[int]], num_vars: int
) -> tuple[bool, Optional[dict], list[str]]:
    steps: list[str] = []
    steps.append("开始 CDCL 求解...")
    result = cdcl(clauses, num_vars, steps)
    if result is not None:
        # 补全未赋值变量
        for v in range(1, num_vars + 1):
            if v not in result:
                result[v] = True
        return True, result, steps
    return False, None, steps


# ─────────────────────────────────────────────
# 3. 文本化 CNF 公式
# ─────────────────────────────────────────────


def cnf_to_text(clauses: list[list[int]]) -> str:
    clause_strs = []
    for clause in clauses:
        lit_strs = [f"¬x{abs(l)}" if l < 0 else f"x{abs(l)}" for l in clause]
        clause_strs.append("(" + " ∨ ".join(lit_strs) + ")")
    return " ∧ ".join(clause_strs)


def assignment_to_text(assignment: dict[int, bool]) -> str:
    parts = [f"x{v}={'真' if val else '假'}" for v, val in sorted(assignment.items())]
    return "，".join(parts)


# ─────────────────────────────────────────────
# 4. 构造 CoT 样本
# ─────────────────────────────────────────────


def build_cot_sample(
    num_vars: int, num_clauses: int, clause_size: int = 3, force_unsat: bool = False
) -> dict:
    if force_unsat:
        clauses = generate_unsat_cnf(num_vars, clause_size)
        num_clauses = len(clauses)
    else:
        clauses = generate_cnf(num_vars, num_clauses, clause_size)
    formula_text = cnf_to_text(clauses)

    is_sat, assignment, steps = solve_cnf(clauses, num_vars)

    # 构造思维链文本
    cot_lines = [
        f"给定合取范式（CNF）公式：{formula_text}",
        f"共 {num_vars} 个变量，{num_clauses} 个子句，每个子句最多 {clause_size} 个文字。",
        "",
        "【求解过程】",
    ]
    cot_lines.extend(steps)
    cot_lines.append("")
    cot_lines.append("【结论】")

    if is_sat:
        answer = f"Assignment: {assignment}"
        cot_lines.append(answer)
    else:
        answer = "UNSAT"
        cot_lines.append(answer)

    text = "\n".join(cot_lines)

    return {
        "text": text,
        "meta": {
            "num_vars": num_vars,
            "num_clauses": num_clauses,
            "clause_size": clause_size,
            "satisfiable": is_sat,
            "answer": answer,
            "clauses": clauses,
        },
    }


# ─────────────────────────────────────────────
# 5. 批量生成 JSONL 数据集
# ─────────────────────────────────────────────


def generate_dataset(
    output_path: str,
    total: int = 1000,
    configs: list[dict] = None,
    seed: int = 42,
    sat_ratio: float = 0.5,  # SAT 样本占比
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(seed)

    if configs is None:
        configs = [
            {"num_vars": 3, "num_clauses": 4, "clause_size": 2, "weight": 0.2},
            {"num_vars": 4, "num_clauses": 6, "clause_size": 3, "weight": 0.3},
            {"num_vars": 5, "num_clauses": 10, "clause_size": 3, "weight": 0.3},
            {"num_vars": 6, "num_clauses": 14, "clause_size": 3, "weight": 0.2},
        ]

    weights = [c["weight"] for c in configs]
    total_weight = sum(weights)
    counts = [int(c["weight"] / total_weight * total) for c in configs]
    counts[-1] += total - sum(counts)

    sat_count = 0
    unsat_count = 0
    samples = []

    for cfg, count in zip(configs, counts):
        n_sat = int(count * sat_ratio)
        n_unsat = count - n_sat

        for _ in range(n_sat):
            # SAT：多次生成直到真的是 SAT
            for _ in range(50):
                s = build_cot_sample(
                    cfg["num_vars"],
                    cfg["num_clauses"],
                    cfg["clause_size"],
                    force_unsat=False,
                )
                if s["meta"]["satisfiable"]:
                    samples.append(s)
                    sat_count += 1
                    break

        for _ in range(n_unsat):
            s = build_cot_sample(
                cfg["num_vars"],
                cfg["num_clauses"],
                cfg["clause_size"],
                force_unsat=True,
            )
            samples.append(s)
            unsat_count += 1

    random.shuffle(samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    actual_total = sat_count + unsat_count
    print(f"✅ 数据集已生成：{output_path}")
    print(f"   总样本数：{actual_total}")
    print(f"   SAT 样本：{sat_count}（{sat_count / actual_total * 100:.1f}%）")
    print(f"   UNSAT 样本：{unsat_count}（{unsat_count / actual_total * 100:.1f}%）")


# ─────────────────────────────────────────────
# 6. 示例：打印一条样本
# ─────────────────────────────────────────────


def print_example():
    sample = build_cot_sample(num_vars=4, num_clauses=6, clause_size=3)
    print("=" * 60)
    print("【示例样本】")
    print("=" * 60)
    print(sample["text"])
    print()
    print("【元数据】")
    print(json.dumps(sample["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 打印一条示例
    print_example()

    # 生成完整数据集
    print("\n" + "=" * 60)
    generate_dataset(
        output_path="./data/cnf_cot_dataset.jsonl",
        total=10,
        seed=42,
    )
