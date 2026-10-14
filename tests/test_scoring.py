import pytest

from quality_eval import (extract_code, gsm8k_extract, gsm8k_gold, humaneval_program, mmlu_extract,
                          run_program, score)

HE = {
    "prompt": 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum of xs."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n",
}


@pytest.mark.parametrize("text,exp", [("B", "B"), ("The answer is C.", "C"), ("Answer: D", "D"),
                                      ("(A) because", "A"), ("none", None)])
def test_mmlu_extract(text, exp):
    assert mmlu_extract(text) == exp


def test_mmlu_score():
    assert score("mmlu", {"answer": 2}, "C", 1)[1]
    assert not score("mmlu", {"answer": 0}, "C", 1)[1]


@pytest.mark.parametrize("text,exp", [
    ("... so 3 + 4 = 7. The answer is 7.", 7.0),
    ("The answer is $1,234.", 1234.0),
    ("The answer is **-5**", -5.0),
    ("We get 12 then 18", 18.0),
    ("no number", None),
])
def test_gsm8k_extract(text, exp):
    assert gsm8k_extract(text) == exp


def test_gsm8k_gold_and_score():
    row = {"answer": "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 1,072"}
    assert gsm8k_gold(row["answer"]) == 1072.0
    assert score("gsm8k", row, "The answer is 1072", 1)[1]
    assert not score("gsm8k", row, "The answer is 1071", 1)[1]


def test_extract_code():
    assert extract_code("x\n```python\ndef f():\n    return 1\n```\ny") == "def f():\n    return 1\n"
    assert extract_code("plain") == "plain"


def test_humaneval_full_function_passes():
    resp = "```python\ndef add(xs):\n    return sum(xs)\n```"
    assert run_program(humaneval_program(HE, resp), 10)


def test_humaneval_body_only_passes():
    assert run_program(humaneval_program(HE, "    return sum(xs)\n"), 10)


def test_humaneval_wrong_fails():
    assert not run_program(humaneval_program(HE, "```python\ndef add(xs):\n    return 0\n```"), 10)


def test_humaneval_timeout_fails():
    assert not run_program(humaneval_program(HE, "```python\ndef add(xs):\n    while True: pass\n```"), 1)
