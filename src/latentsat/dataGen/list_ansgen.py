import json
import os
import random
from collections import OrderedDict

from latentsat.dataGen.genCoT import generate_cnf, generate_unsat_cnf, solve_cnf


def build_sample(
    num_vars: int, num_clauses: int, clause_size: int = 3, force_unsat: bool = False
) -> dict:
    if force_unsat:
        clauses = generate_unsat_cnf(num_vars, clause_size)
        num_clauses = len(clauses)
    else:
        clauses = generate_cnf(num_vars, num_clauses, clause_size)

    is_sat, assignment, _ = solve_cnf(clauses, num_vars)

    if is_sat:
        assignment[0] = True
        assignment = OrderedDict(sorted(assignment.items()))
        answer = [int(assignment[i]) for i in assignment.keys()]
    else:
        answer = [int(False)]

    return {
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "clause_size": clause_size,
        "satisfiable": is_sat,
        "answer": answer,
        "clauses": clauses,
    }


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
                s = build_sample(
                    cfg["num_vars"],
                    cfg["num_clauses"],
                    cfg["clause_size"],
                    force_unsat=False,
                )
                if s["satisfiable"]:
                    samples.append(s)
                    sat_count += 1
                    break

        for _ in range(n_unsat):
            s = build_sample(
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


if __name__ == "__main__":
    generate_dataset(
        output_path="data/list_data.jsonl",
        total=1000,
        configs=None,
        seed=42,
        sat_ratio=0.7,
    )
