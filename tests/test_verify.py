import json

import pytest

from latentsat.utils.verify import extract_conclusion, verify


@pytest.fixture
def data():
    with open("./data/clean_data.jsonl", "r") as f:
        data = f.readlines()
    list_dict = map(json.loads, data)
    return list(list_dict)


def test_verify(data):
    for i in data:
        satisfiable = i["satisfiable"]
        clauses = i["clauses"]
        answer = i["answer"]
        assignment = extract_conclusion(answer)
        print(assignment)
        assert verify(clauses, assignment, satisfiable)
