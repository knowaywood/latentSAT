import ast


def is_SAT(text: str) -> bool:
    if "UNSAT" in text:
        return False
    else:
        return True


def extract_conclusion(text: str) -> dict[int, bool]:
    if is_SAT(text):
        text = text.split("Assignment:", 1)[-1].strip()
        return ast.literal_eval(text)
    else:
        return {}


def verify(
    clauses: list[list[int]], assignment: dict[int, bool], satisfiable: bool
) -> bool:
    if assignment == {} and satisfiable:
        return False
    if assignment == {} and not satisfiable:
        return True
    if assignment != {} and not satisfiable:
        return False
    for clause in clauses:
        vec = []
        for j in clause:
            if j < 0:
                vec.append(int(not assignment[-j]))
            else:
                vec.append(int(assignment[j]))
        if sum(vec) == 0:
            return False
    return True


if __name__ == "__main__":
    text = "Assignment: {1: True, 2: True, 4: False, 3: True, 5: False}"
    clauses = [
        [-5, 3, 2],
        [-4, 1, -2],
        [-4, -3, -2],
        [-2, -1, -4],
        [3, 4, -1],
        [-5, 4, 2],
        [-2, 5, 3],
        [1, -5, -2],
        [-1, 4, -5],
        [-3, -1, -5],
    ]
    print(verify(clauses, extract_conclusion(text), True))
