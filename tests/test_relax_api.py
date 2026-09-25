"""POST /api/v1/chains/relax 的 API 层测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload():
    return {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 10, "upper": 12},
            {"id": "c2", "start": "B", "end": "C", "lower": 5, "upper": 8},
        ],
    }


def test_relax_feasible_request_e_zero():
    r = client.post("/api/v1/chains/relax", json=base_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "feasible"
    assert body["unit"] == "um"
    assert body["relaxation"] == 0
    assert body["coordinates"] == {"A": 0, "B": 12, "C": 20}
    # e=0：放宽区间与原区间一致，按请求顺序逐条给出
    assert body["constraints"] == [
        {
            "id": "c1", "start": "A", "end": "B",
            "original_lower": 10, "original_upper": 12,
            "relaxed_lower": 10, "relaxed_upper": 12,
        },
        {
            "id": "c2", "start": "B", "end": "C",
            "original_lower": 5, "original_upper": 8,
            "relaxed_lower": 5, "relaxed_upper": 8,
        },
    ]


def test_relax_infeasible_chain_returns_min_e_and_intervals():
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 10},
            {"id": "c2", "start": "B", "end": "C", "lower": 9, "upper": 10},
            {"id": "c3", "start": "C", "end": "A", "lower": -1, "upper": 0},
        ],
    }
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["relaxation"] == 6  # ceil(17/3)
    coords = body["coordinates"]
    assert coords["A"] == 0
    # 同一组整数坐标满足放宽后全部约束
    for it in body["constraints"]:
        diff = coords[it["end"]] - coords[it["start"]]
        assert it["relaxed_lower"] <= diff <= it["relaxed_upper"]
        assert it["relaxed_lower"] == it["original_lower"] - 6
        assert it["relaxed_upper"] == it["original_upper"] + 6
    # 逐条区间按请求顺序回显
    assert [it["id"] for it in body["constraints"]] == ["c1", "c2", "c3"]


def test_relax_covers_multiple_conflicting_cycles():
    # 两个只共用 anchor 的独立矛盾环：B 环需 2、C 环需 4，e 必须取 4
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 9},
            {"id": "c2", "start": "A", "end": "B", "lower": 5, "upper": 5},
            {"id": "c3", "start": "A", "end": "C", "lower": 10, "upper": 10},
            {"id": "c4", "start": "A", "end": "C", "lower": 3, "upper": 3},
        ],
    }
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["relaxation"] == 4
    coords = body["coordinates"]
    for it in body["constraints"]:
        diff = coords[it["end"]] - coords[it["start"]]
        assert it["relaxed_lower"] <= diff <= it["relaxed_upper"]


def test_relax_datum_points_stay_zero():
    payload = {
        "datum_points": ["D1", "D2"],
        "measure_points": ["M"],
        "constraints": [
            {"id": "c1", "start": "D1", "end": "D2", "lower": 1, "upper": 2},
            {"id": "c2", "start": "D1", "end": "M", "lower": 0, "upper": 5},
        ],
    }
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["relaxation"] == 1
    assert body["coordinates"]["D1"] == 0
    assert body["coordinates"]["D2"] == 0
    c1 = body["constraints"][0]
    assert (
        c1["original_lower"] == 1
        and c1["original_upper"] == 2
        and c1["relaxed_lower"] == 0
        and c1["relaxed_upper"] == 3
    )


def test_relax_uses_chain_request_validation():
    # 沿用 ChainRequest：未连通、反向区间走同一套 422
    payload = {
        "datum_points": ["A"],
        "measure_points": ["B", "C"],
        "constraints": [
            {"id": "c1", "start": "A", "end": "B", "lower": 1, "upper": 2}
        ],
    }
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "point_not_connected_to_datum"

    payload["constraints"][0] = {
        "id": "c1", "start": "A", "end": "B", "lower": 9, "upper": 1
    }
    payload["measure_points"] = ["B"]
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"][0]["code"] == "reversed_interval"


def test_relax_strict_integer_validation_422():
    payload = base_payload()
    payload["constraints"][0]["lower"] = 9.5
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 422


def test_relax_rejects_extra_field_422():
    payload = base_payload()
    payload["bogus"] = 1
    r = client.post("/api/v1/chains/relax", json=payload)
    assert r.status_code == 422


def test_relax_openapi_documented():
    schema = client.get("/openapi.json").json()
    assert "/api/v1/chains/relax" in schema["paths"]
    assert "RelaxResponse" in schema["components"]["schemas"]
    assert "RelaxedConstraint" in schema["components"]["schemas"]
