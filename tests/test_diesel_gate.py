import pytest

from serving_app.diesel_gate import WEEK1_RMSE_MAX, check_gate

NAIVE = [30.0, 60.0, 80.0, 90.0]


def test_passes_when_all_conditions_hold():
    assert check_gate([27.0, 53.0, 68.0, 76.0], NAIVE) == {"passed": True, "reasons": []}


@pytest.mark.parametrize(
    ("week1", "passed"), [(WEEK1_RMSE_MAX, True), (WEEK1_RMSE_MAX + 0.01, False)]
)
def test_week1_absolute_limit_boundary(week1, passed):
    naive = [100.0, 100.0, 100.0, 100.0]
    assert check_gate([week1, 60.0, 70.0, 80.0], naive)["passed"] is passed


def test_tie_with_naive_fails_on_any_week():
    result = check_gate([27.0, 53.0, 80.0, 76.0], NAIVE)  # 3주차가 naive와 같음
    assert not result["passed"]
    assert result["reasons"] == ["3주차 RMSE 80.00 >= naive 80.00"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_fail(bad):
    assert not check_gate([bad, 53.0, 68.0, 76.0], NAIVE)["passed"]
    assert not check_gate([27.0, 53.0, 68.0, 76.0], [bad, 60.0, 80.0, 90.0])["passed"]
    assert not check_gate([27.0, 53.0, 68.0, 76.0], NAIVE, [bad, 1.0, 1.0, 1.0])["passed"]


def test_production_comparison_on_mean_of_weeks():
    model = [27.0, 53.0, 68.0, 76.0]  # 평균 56.0
    assert check_gate(model, NAIVE, [28.0, 52.0, 68.0, 76.0])["passed"]  # 평균 동점은 교체 허용
    # 1주차가 Production보다 나빠도 평균이 좋으면 통과 (한 주차 운에 흔들리지 않게)
    assert check_gate(model, NAIVE, [26.0, 55.0, 70.0, 78.0])["passed"]
    worse = check_gate(model, NAIVE, [27.0, 53.0, 68.0, 75.6])
    assert worse == {"passed": False, "reasons": ["1~4주 평균 RMSE 56.00 > Production 55.90"]}
    assert not check_gate(model, NAIVE, [27.0, 53.0])["passed"]


def test_empty_or_mismatched_inputs_fail():
    assert not check_gate([], [])["passed"]
    assert not check_gate([27.0, 53.0], NAIVE)["passed"]
