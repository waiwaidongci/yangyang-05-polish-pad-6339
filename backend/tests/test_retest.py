"""Retest and duplicate handling for the polishing-cloth comparison.

Run from the backend directory: python -m tests.test_retest
"""

from datetime import date, timedelta

from src.domain import analysis


def _lot(lot_code):
    return [row for row in analysis.seed() if row["lot_code"] == lot_code]


def _piece_states(result, piece):
    return [(item["measured_on"], item["roughness_nm"], item["version_state"], item["note"])
            for item in result["versions"]["rows"] if item["piece_code"] == piece]


def _with_ids(rows):
    return [{**row, "_observation_id": index + 1} for index, row in enumerate(rows)]


def test_baseline_is_unchanged():
    result = analysis.analyze(analysis.seed(), {"lot_code": "GD-261109-A"})
    metrics = {item["label"]: item["value"] for item in result["metrics"]}
    assert metrics["独立工件数"] == 60
    assert metrics["并列校正 H"] == 22.891633
    assert metrics["隔离工件数"] == 0
    assert result["rows"][0]["median_nm"] == 22.0
    assert result["rows"][0]["reduction_nm"] == 4.5


def test_same_batch_retest_is_accepted_and_latest_date_wins():
    rows = _with_ids(_lot("GD-261109-A"))
    original = rows[0]
    retest = {**original,
              "_observation_id": 1000,
              "measured_on": (date.fromisoformat(original["measured_on"]) + timedelta(days=4)).isoformat(),
              "roughness_nm": original["roughness_nm"] - 5}
    result = analysis.analyze(rows + [retest], {"lot_code": "GD-261109-A"})
    states = _piece_states(result, original["piece_code"])
    assert [state[2] for state in states] == ["被返测替代", "采用"]
    assert states[-1][0] == retest["measured_on"]
    assert result["versions"]["quarantined"] == []
    assert sum(row["n"] for row in result["rows"]) == 60


def test_import_order_does_not_decide():
    base = _lot("GD-261109-A")
    piece = base[0]["piece_code"]
    newer = {**base[0], "measured_on": "2026-11-12", "roughness_nm": 20}
    older = {**base[0], "measured_on": "2026-11-10", "roughness_nm": 40}
    result = analysis.analyze(base + [newer, older], {"lot_code": "GD-261109-A"})
    adopted = [state for state in _piece_states(result, piece) if state[2] == "采用"]
    assert [(state[0], state[1]) for state in adopted] == [("2026-11-12", 20)]


def test_conflicting_latest_reading_quarantines_only_that_piece():
    base = _with_ids(_lot("GD-261109-A"))
    disputed = base[1]["piece_code"]
    first = {**base[1], "_observation_id": 1001, "measured_on": "2026-11-20", "roughness_nm": 21}
    second = {**base[1], "_observation_id": 1002, "measured_on": "2026-11-20", "roughness_nm": 30}
    result = analysis.analyze(base + [first, second], {"lot_code": "GD-261109-A"})
    quarantined = result["versions"]["quarantined"]
    assert len(quarantined) == 1
    assert quarantined[0]["piece_code"] == disputed
    assert "2026-11-20" in quarantined[0]["reason"]
    assert "21" in quarantined[0]["reason"] and "30" in quarantined[0]["reason"]
    assert all(state[2] == "隔离冲突" for state in _piece_states(result, disputed))
    assert sum(row["n"] for row in result["rows"]) == 59
    assert result["rows"][0]["median_nm"] == 22.0


def test_identical_same_day_duplicate_is_not_conflict():
    base = _lot("GD-261109-A")
    duplicate = dict(base[0])
    result = analysis.analyze(base + [duplicate], {"lot_code": "GD-261109-A"})
    assert result["versions"]["quarantined"] == []
    states = [state[2] for state in _piece_states(result, base[0]["piece_code"])]
    assert sorted(states) == ["同日重复", "采用"]
    assert sum(row["n"] for row in result["rows"]) == 60


def test_every_version_is_traceable_to_a_record():
    rows = _with_ids(analysis.seed())
    result = analysis.analyze(rows, {"lot_code": "GD-261116-B"})
    versions = result["versions"]["rows"]
    assert len(versions) == 36
    assert {item["observation_id"] for item in versions} == {row["_observation_id"] for row in rows
                                                             if row["lot_code"] == "GD-261116-B"}


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print("ok", name)
