"""solve_relax 的随机差分测试：与逐 e 暴力枚举交叉验证最小放宽量。

对随机实例从 e=0 起逐一代入放宽约束，用 test_random.brute_force_solutions
枚举整数坐标判定可行性，首个可行 e 即暴力 e*，与 Karp 最大平均环求得的
relaxation 比对，并核对返回坐标满足放宽后全部约束、基准点为 0；同时覆盖
单基准点与多基准点（基准收缩）两类实例。
"""

from __future__ import annotations

import random

from app.schemas import ChainRequest, ConstraintIn
from app.solver import solve_relax
from tests.test_random import random_instance
from tests.test_reconcile_random import _multi_datum_instance


def brute_force_relaxation(datum, measure, constraints) -> int:
    """逐 e 放宽，每个 e 下在受限整数坐标范围内暴力找可行赋值。"""
    from tests.test_random import brute_force_solutions

    e = 0
    while True:
        widened = [
            ConstraintIn(
                id=c.id,
                start=c.start,
                end=c.end,
                lower=c.lower - e,
                upper=c.upper + e,
            )
            for c in constraints
        ]
        if next(brute_force_solutions(datum, measure, widened), None) is not None:
            return e
        e += 1


def _assert_relax_matches(datum, measure, constraints) -> None:
    req_obj = ChainRequest(
        datum_points=datum, measure_points=measure, constraints=constraints
    )
    res = solve_relax(req_obj)
    e_star = brute_force_relaxation(datum, measure, constraints)
    assert res.feasible
    assert res.relaxation == e_star, (
        f"e mismatch: got {res.relaxation}, bf {e_star}, "
        f"constraints={[(c.start, c.end, c.lower, c.upper) for c in constraints]}"
    )
    coords = res.coordinates
    assert all(coords[d] == 0 for d in datum)
    for c in constraints:
        assert c.lower - e_star <= coords[c.end] - coords[c.start] <= c.upper + e_star
        assert isinstance(coords[c.end], int)


def test_random_relax_matches_brute_force():
    rng = random.Random(20260925)
    zero_cases = 0
    positive_cases = 0
    for _ in range(250):
        datum, measure, constraints = random_instance(rng)
        _assert_relax_matches(datum, measure, constraints)
        res = solve_relax(
            ChainRequest(
                datum_points=datum,
                measure_points=measure,
                constraints=constraints,
            )
        )
        if res.relaxation == 0:
            zero_cases += 1
        else:
            positive_cases += 1
    # 随机实例既应覆盖原请求可行（e=0），也应覆盖需要放宽（e>0）的冲突环
    assert zero_cases > 0 and positive_cases > 0


def test_random_relax_multiple_datums_match_brute_force():
    rng = random.Random(7717)
    for _ in range(150):
        datum, measure, constraints = _multi_datum_instance(rng)
        _assert_relax_matches(datum, measure, constraints)
