import ast


class VerifyText:
    def _is_sat(self, text: str) -> bool:
        if "UNSAT" in text:
            return False
        else:
            return True

    def extract_conclusion(self, text: str) -> dict[int, bool]:
        if self._is_sat(text):
            text = text.split("Assignment:", 1)[-1].strip()
            return ast.literal_eval(text)
        else:
            return {}

    def verify(
        self,
        clauses: list[list[int]],
        num_vars: int,
        assignment: dict[int, bool],
        satisfiable: bool,
    ) -> bool:
        if len(assignment) != num_vars and satisfiable:
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

    def __call__(
        self,
        clauses: list[list[int]],
        num_vars: int,
        assignment: dict[int, bool],
        satisfiable: bool,
    ) -> bool:
        return self.verify(clauses, num_vars, assignment, satisfiable)


class VerifyList:
    def _is_sat(self, list: list[int]) -> bool:
        if list[0] == 0:
            return False
        elif list[0] == 1:
            return True
        else:
            assert Exception("invalid list")

    def verify_sat(
        self,
        assignment: list[int],
        satisfiable: bool,
    ):
        if satisfiable and assignment[0] == 1:
            return True
        elif satisfiable and assignment[0] == 0:
            return False
        elif not satisfiable and assignment[0] == 1:
            return False
        elif not satisfiable and assignment[0] == 0:
            return True
        else:
            return False

    def verify_len_err(self, num_vars: int, assignment: list[int], satisfiable: bool):
        if satisfiable and num_vars + 1 != len(assignment):
            return False
        if not satisfiable and 1 != len(assignment):
            return False
        return True

    def verify(
        self,
        clauses: list[list[int]],
        num_vars: int,
        assignment: list[int],
        satisfiable: bool,
    ) -> bool:
        if assignment[0] == 0 and not satisfiable:
            return True
        if assignment[0] != 0 and not satisfiable:
            return False
        if num_vars + 1 > len(assignment) and satisfiable:
            return False
        for clause in clauses:
            is_clause_sat = any(
                self._not(assignment[-j]) if j < 0 else assignment[j] for j in clause
            )
            if not is_clause_sat:
                return False
        return True

    def _not(self, a: int):
        if a == 0:
            return 1
        elif a == 1:
            return 0
        else:
            assert Exception("invalid int")

    def __call__(
        self,
        clauses: list[list[int]],
        num_vars: int,
        assignment: list[int],
        satisfiable: bool,
    ) -> bool:
        return self.verify(clauses, num_vars, assignment, satisfiable)


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
