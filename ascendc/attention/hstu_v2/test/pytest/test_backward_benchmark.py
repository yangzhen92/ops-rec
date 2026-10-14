"""Official 74-case HSTU V2 backward performance matrix."""

import pytest

from utils.user_case_benchmark import UserCaseRunner, pytest_params, user_cases

pytestmark = pytest.mark.benchmark
CASES = user_cases()
PARAMS = pytest_params()


@pytest.fixture(scope="session")
def runner():
    return UserCaseRunner("backward")


@pytest.mark.parametrize("case", PARAMS)
def test_backward_user_case(case, runner):
    runner.run(case)


def test_backward_user_case_coverage():
    assert len(CASES) == 74
    assert sum(case.family == "user_case_1" for case in CASES) == 10
    assert sum(case.family == "user_case_2" for case in CASES) == 64
