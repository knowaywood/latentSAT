import json

import pytest

from latentsat.utils.verify import VerifyList, VerifyText


@pytest.fixture
def data():
    with open("./data/clean_data.jsonl", "r") as f:
        data = f.readlines()
    list_dict = map(json.loads, data)
    return list(list_dict)


def test_verify_text(data):
    verify = VerifyText()
    for i in data:
        satisfiable = i["satisfiable"]
        clauses = i["clauses"]
        answer = i["answer"]
        num_vars = i["num_vars"]

        assignment = verify.extract_conclusion(answer)
        assert verify(clauses, num_vars,assignment, satisfiable)


@pytest.fixture
def list_data():
    with open("./data/list_data.jsonl", "r") as f:
        data = f.readlines()
    list_dict = map(json.loads, data)
    return list(list_dict)


def test_verify_list(list_data):
    verify = VerifyList()
    for i in list_data:
        num_vars = i["num_vars"]
        satisfiable = i["satisfiable"]
        clauses = i["clauses"]
        answer = i["answer"]
        assert verify(clauses, num_vars, answer, satisfiable)
