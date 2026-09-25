"""统一放宽（solve_relax）求解器单元测试。"""

from __future__ import annotations

from app.schemas import ChainRequest
from app.solver import solve_relax
from tests.test_solver import C


def req(datum, measure, constraints) -> ChainRequest:
    return ChainRequest(
        datum_points=list(datum),
        measure_points=list(measure),
        constraints=list(constraints),
    )


def check_solution(r: ChainRequest, res, e: int) -> None:
    """断言返回 e 与坐标满足放宽后全部约束，且 e−1 已不可行。"""
    assert res.feasible
    assert res.relaxation == e
    coords = res.coordinates
    for d in r.datum_points:
        assert coords[d] == 0  # 基准点固定为 0，不参与放宽
    for c in r.constraints:
        diff = coords[c.end] - coords[c.start]
        assert c.lower - e <= diff <= c.upper + e
        assert isinstance(diff, int)
    # 逐条原区间 / 放宽区间按请求顺序给出
    assert [it.cid for it in res.relaxed_constraints] == [
        c.id for c in r.constraints
    ]
    for it, c in zip(res.relaxed_constraints, r.constraints):
        assert (it.start, it.end) == (c.start, c.end)
        assert it.original_lower == c.lower and it.original_upper == c.upper
        assert it.relaxed_lower == c.lower - e
        assert it.relaxed_upper == c.upper + e
    # 最小性：e-1 放宽量下整条链仍不可行（对放宽 e-1 的新请求做 /check 校核）
    if e > 0:
        from app.solver import solve

        smaller = req(
            r.datum_points,
            r.measure_points,
            [
                C(c.id, c.start, c.end, c.lower - (e - 1), c.upper + (e - 1))
                for c in r.constraints
            ],
        )
        assert not solve(smaller).feasible


# ---------- 原请求可行：e = 0 ----------

def test_feasible_request_needs_no_relaxation():
    r = req(
        ["A"], ["B", "C"],
        [C("c1", "A", "B", 10, 12), C("c2", "B", "C", 5, 8)],
    )
    res = solve_relax(r)
    check_solution(r, res, 0)
    # e=0 时放宽区间就是原区间，坐标与 /check 的最大可行解一致
    assert res.coordinates == {"A": 0, "B": 12, "C": 20}


def test_zero_width_exact_constraints_e_zero():
    r = req(["A"], ["B"], [C("c1", "A", "B", 5, 5)])
    check_solution(r, solve_relax(r), 0)


def test_negative_bounds_feasible_e_zero():
    r = req(["A"], ["B"], [C("c1", "A", "B", -12, -10)])
    res = solve_relax(r)
    check_solution(r, res, 0)
    assert res.coordinates["B"] == -10


# ---------- 单冲突环 ----------

def test_classic_three_edge_contradiction():
    # W = 10 + 10 + (-1) ... 三条环边：upper=10、upper=10、-lower=1（c3 C→A [−1,0]）
    # 环权 10+10+1 = 21 取反向矛盾环：lower 边 -9、-9、upper 边 0 ⇒ W=-17，k=3
    # e* = ceil(17/3) = 6
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 10),
            C("c2", "B", "C", 9, 10),
            C("c3", "C", "A", -1, 0),
        ],
    )
    check_solution(r, solve_relax(r), 6)


def test_exact_chain_loop_violation_rounds_up():
    # B=9（精确）、C=B+9（精确）⇒ C=18，而 c3 要求 A-C ∈ [0,0]：
    # 纯 lower/upper 矛盾环 W=-18，k=3 ⇒ e* = 6
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 9),
            C("c2", "B", "C", 9, 9),
            C("c3", "A", "C", 0, 0),
        ],
    )
    check_solution(r, solve_relax(r), 6)


def test_self_loop_contradiction():
    # coord(A)-coord(A) ∈ [1,2]：自环 W=-1，k=1 ⇒ e*=1
    r = req(
        ["A"], ["B"],
        [C("s", "A", "A", 1, 2), C("c1", "A", "B", 1, 2)],
    )
    check_solution(r, solve_relax(r), 1)


def test_datum_conflict_relaxes_constraint_not_pin():
    # 两基准点必须同为 0，c1 却要求 D2-D1 ∈ [1,2]：矛盾环含 1 条约束边，
    # pin 边不放宽 ⇒ e*=1；放宽后 c1 为 [0,3] 即可容纳 0 差
    r = req(
        ["D1", "D2"], ["M"],
        [C("c1", "D1", "D2", 1, 2), C("c2", "D1", "M", 0, 5)],
    )
    res = solve_relax(r)
    check_solution(r, res, 1)
    assert res.coordinates["D1"] == 0 and res.coordinates["D2"] == 0


# ---------- 多个冲突环必须同时覆盖 ----------

def test_two_independent_loops_require_max_not_first_cycle():
    # A-B：c1 要求 B=9、c2 要求 B=5（平行同向精确约束），2 边环 W=-4 ⇒ e≥2；
    # A-C：c3 要求 C=10、c4 要求 C=3，2 边环 W=-7 ⇒ e≥4。
    # 两环只共用 anchor，首个矛盾环只给 2，全局最小 e 为 4。
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 9),
            C("c2", "A", "B", 5, 5),
            C("c3", "A", "C", 10, 10),
            C("c4", "A", "C", 3, 3),
        ],
    )
    res = solve_relax(r)
    check_solution(r, res, 4)
    # 返回坐标同时满足两个放宽后的环
    b = res.coordinates["B"]
    c = res.coordinates["C"]
    assert 9 - 4 <= b - 0 <= 9 + 4 and 5 - 4 <= b - 0 <= 5 + 4
    assert 10 - 4 <= c - 0 <= 10 + 4 and 3 - 4 <= c - 0 <= 3 + 4


def test_distinct_cycles_different_k_rounding_max():
    # B 环：B-A ∈ [9,10] 与 B-A ∈ [-1,0] 冲突（2 条约束边），
    # 最矛盾 2 边环 W=-9 ⇒ e≥5；
    # C 环：C 必须同时为 10 与 3（2 边环 W=-7）⇒ e≥4。取最大 5。
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 10),
            C("c2", "B", "A", 0, 0),
            C("c3", "A", "C", 10, 10),
            C("c4", "A", "C", 3, 3),
        ],
    )
    check_solution(r, solve_relax(r), 5)


def test_relaxation_is_uniform_all_constraints_widened():
    # 即使是未直接参与矛盾环的约束也按同一 e 放宽（区间端点如实回显）
    r = req(
        ["A"], ["B", "C"],
        [
            C("c1", "A", "B", 9, 9),
            C("c2", "A", "B", 5, 5),  # B 矛盾环 ⇒ e=2
            C("c3", "B", "C", 100, 200),  # 宽松约束，同样 ±e
        ],
    )
    res = solve_relax(r)
    assert res.relaxation == 2
    by_id = {it.cid: it for it in res.relaxed_constraints}
    assert (by_id["c3"].relaxed_lower, by_id["c3"].relaxed_upper) == (98, 202)
    check_solution(r, res, 2)


# ---------- 多基准点 ----------

def test_multiple_datums_all_zero_in_coordinates():
    r = req(
        ["D1", "D2"], ["M"],
        [C("c1", "D1", "M", 3, 7), C("c2", "M", "D2", -7, -3)],
    )
    res = solve_relax(r)
    check_solution(r, res, 0)
    assert res.coordinates["D1"] == 0 and res.coordinates["D2"] == 0


# ---------- 静态校验沿用 ChainRequest ----------

def test_static_validation_reused():
    import pytest

    from app.solver import ValidationError

    # 未连通 / 反向区间 / 未知点：与 /check 相同的 422 领域错误
    r = req(["A"], ["B", "C"], [C("c1", "A", "B", 1, 2)])
    with pytest.raises(ValidationError) as ei:
        solve_relax(r)
    assert {e["code"] for e in ei.value.errors} == {
        "point_not_connected_to_datum"
    }

    r2 = req(["A"], ["B"], [C("c1", "A", "B", 9, 1)])
    with pytest.raises(ValidationError) as ei:
        solve_relax(r2)
    assert ei.value.errors[0]["code"] == "reversed_interval"
