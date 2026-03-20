from typing import Optional


def cdcl(
    clauses: list[list[int]], num_vars: int, steps: list[str]
) -> Optional[dict[int, bool]]:
    working_clauses = [list(c) for c in clauses]
    assignment: dict[int, bool] = {}
    levels: dict[int, int] = {}
    reasons: dict[int, Optional[list[int]]] = {}
    trail: list[int] = []
    current_level = 0

    def lit_is_true(lit: int) -> bool:
        var = abs(lit)
        if var not in assignment:
            return False
        return assignment[var] if lit > 0 else (not assignment[var])

    def pick_branch_var() -> Optional[int]:
        for v in range(1, num_vars + 1):
            if v not in assignment:
                return v
        return None

    def enqueue(lit: int, level: int, reason: Optional[list[int]]) -> bool:
        var = abs(lit)
        val = lit > 0
        if var in assignment:
            return assignment[var] == val
        assignment[var] = val
        levels[var] = level
        reasons[var] = reason
        trail.append(lit)
        return True

    def unit_propagate() -> Optional[list[int]]:
        changed = True
        while changed:
            changed = False
            for clause in working_clauses:
                if any(lit_is_true(l) for l in clause):
                    continue
                unassigned = [l for l in clause if abs(l) not in assignment]
                if not unassigned:
                    return clause
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if not enqueue(lit, current_level, clause):
                        return clause
                    steps.append(
                        f"📌 [L{current_level}] 单元传播：x{abs(lit)} = {lit > 0}（来自子句 {clause}）"
                    )
                    changed = True
        return None

    def backtrack(target_level: int):
        nonlocal trail
        kept: list[int] = []
        for lit in trail:
            var = abs(lit)
            if levels[var] <= target_level:
                kept.append(lit)
            else:
                del assignment[var]
                del levels[var]
                del reasons[var]
        trail = kept

    def analyze_conflict(conflict_clause: list[int]) -> tuple[list[int], int]:
        learned = list(dict.fromkeys(conflict_clause))

        def count_current_level_lits(clause: list[int]) -> int:
            return sum(1 for l in clause if levels.get(abs(l), -1) == current_level)

        while count_current_level_lits(learned) > 1:
            curr_lits = [l for l in learned if levels.get(abs(l), -1) == current_level]
            if not curr_lits:
                break
            latest_lit = None
            for t_lit in reversed(trail):
                if abs(t_lit) in {abs(l) for l in curr_lits}:
                    latest_lit = next(l for l in curr_lits if abs(l) == abs(t_lit))
                    break
            if latest_lit is None:
                break
            reason_clause = reasons.get(abs(latest_lit))
            if reason_clause is None:
                break
            pivot = -latest_lit
            combined = [l for l in learned if l != latest_lit]
            combined.extend(l for l in reason_clause if l != pivot)
            learned = list(dict.fromkeys(combined))

        backjump_level = 0
        for lit in learned:
            lv = levels.get(abs(lit), 0)
            if lv != current_level:
                backjump_level = max(backjump_level, lv)
        return learned, backjump_level

    while True:
        conflict = unit_propagate()
        if conflict is not None:
            if current_level == 0:
                steps.append(f"❌ 在决策层 0 发生冲突：{conflict}，判定 UNSAT")
                return None

            learned_clause, backjump_level = analyze_conflict(conflict)
            working_clauses.append(learned_clause)
            steps.append(f"⚠️ 冲突分析：学习子句 {learned_clause}")
            steps.append(f"↩️ 非时间顺序回跳：L{current_level} -> L{backjump_level}")

            backtrack(backjump_level)
            current_level = backjump_level
            continue

        all_satisfied = all(
            any(lit_is_true(l) for l in clause) for clause in working_clauses
        )
        if all_satisfied:
            steps.append("✅ 所有子句均满足，找到可满足赋值")
            return assignment

        var = pick_branch_var()
        if var is None:
            return assignment
        current_level += 1
        decision_lit = var
        enqueue(decision_lit, current_level, None)
        steps.append(f"🔀 [L{current_level}] 决策：尝试 x{var} = true")
