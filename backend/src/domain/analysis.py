"""Compare three polishing cloths on independent workpieces from a single lot."""

from datetime import date, timedelta
from itertools import groupby
from math import exp, fsum
from operator import itemgetter
from statistics import median, quantiles

CURRENT_CLOTH = "短绒布（现用）"
CLOTHS = (CURRENT_CLOTH, "致密无纺布", "微孔聚氨酯布")
DEFAULT_LOT = "GD-261109-A"
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_REDUCTION = 3
MIN_GROUP_SIZE = 5

# In each lot, the three series represent separate workpieces. The gauge records
# whole nanometres, so equal readings occur both within and across cloth groups.
LOT_READINGS = (
    (
        DEFAULT_LOT,
        date(2026, 11, 9),
        (
            (27, 24, 29, 25, 33, 22, 26, 29, 25, 28, 24, 34, 26, 23, 31, 27, 30, 26, 24, 28),
            (23, 25, 24, 26, 27, 22, 23, 31, 25, 24, 26, 20, 24, 23, 29, 28, 25, 26, 21, 24),
            (21, 19, 24, 22, 22, 20, 25, 18, 23, 22, 20, 26, 21, 19, 24, 21, 23, 22, 27, 20),
        ),
    ),
    (
        "GD-261116-B",
        date(2026, 11, 16),
        (
            (26, 24, 27, 28, 23, 25, 29, 24, 27, 26, 25, 28),
            (25, 27, 23, 26, 28, 24, 26, 30, 25, 27, 24, 28),
            (24, 25, 27, 23, 28, 26, 25, 27, 24, 28, 26, 27),
        ),
    ),
)


def seed() -> list[dict]:
    rows = []
    for lot, first_day, readings in LOT_READINGS:
        serial = 1
        for index, values in enumerate(zip(*readings, strict=True)):
            measured_on = (first_day + timedelta(days=index // 4)).isoformat()
            # Rotate the order of the three cloths across consecutive workpieces.
            for offset in range(len(CLOTHS)):
                cloth_index = (index + offset) % len(CLOTHS)
                rows.append({
                    "lot_code": lot,
                    "piece_code": f"{lot}-{serial:03d}",
                    "measured_on": measured_on,
                    "cloth_type": CLOTHS[cloth_index],
                    "roughness_nm": values[cloth_index],
                })
                serial += 1
    return rows


def validate(row: dict) -> None:
    for key, label in (("lot_code", "工件批号"), ("piece_code", "工件编号")):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}不能为空。")
    measured_on = row.get("measured_on")
    try:
        if date.fromisoformat(measured_on).isoformat() != measured_on:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("测量日期须为有效的 YYYY-MM-DD 日期。") from None
    if row.get("cloth_type") not in CLOTHS:
        raise ValueError("请选择现用短绒布、致密无纺布或微孔聚氨酯布。")
    roughness = row.get("roughness_nm")
    if (
        type(roughness) not in (int, float)
        or not 1 <= roughness <= 1000
        or int(roughness) != roughness
    ):
        raise ValueError("抛光后粗糙度 Ra 须为 1 至 1000 nm 的整数。")


def _lot_groups(rows: list[dict], lot: str) -> dict[str, list[int]]:
    groups = {cloth: [] for cloth in CLOTHS}
    pieces = set()
    for row in rows:
        row_lot = row.get("lot_code")
        if not isinstance(row_lot, str) or row_lot.strip() != lot:
            continue
        validate(row)
        piece = row["piece_code"].strip()
        if piece in pieces:
            raise ValueError(f"批次 {lot} 的工件 {piece} 重复。每件工件只保留一个布型和一次读数。")
        pieces.add(piece)
        groups[row["cloth_type"]].append(int(row["roughness_nm"]))
    if not pieces:
        raise ValueError(f"批次 {lot} 没有可分析的工件记录。")
    short = [f"{cloth} {len(values)} 件" for cloth, values in groups.items() if len(values) < MIN_GROUP_SIZE]
    if short:
        raise ValueError(f"每种抛光布至少需要 {MIN_GROUP_SIZE} 件独立工件。当前不足：{'、'.join(short)}。")
    return groups


def _kruskal_wallis(groups: dict[str, list[int]]) -> tuple[float, float, float, dict[str, float]]:
    ranked = sorted((value, cloth) for cloth, values in groups.items() for value in values)
    total = len(ranked)
    rank_sums = dict.fromkeys(groups, 0.0)
    tied_term = 0
    first_rank = 1
    for _, tied in groupby(ranked, key=itemgetter(0)):
        members = list(tied)
        size = len(members)
        mean_rank = first_rank + (size - 1) / 2
        for _, cloth in members:
            rank_sums[cloth] += mean_rank
        tied_term += size**3 - size
        first_rank += size

    denominator = total**3 - total
    if tied_term == denominator:
        raise ValueError("该批次的粗糙度读数完全相同，无法计算秩检验。请核对量具分辨率和读数。")
    tie_correction = (denominator - tied_term) / denominator
    mean_ranks = {cloth: rank_sums[cloth] / len(values) for cloth, values in groups.items()}
    center = (total + 1) / 2
    dispersion = fsum(len(groups[cloth]) * (rank - center)**2 for cloth, rank in mean_ranks.items())
    statistic = 12 * dispersion / (total * (total + 1) * tie_correction)
    # Three required groups give two degrees of freedom. The chi-square
    # survival function at two degrees of freedom is exp(-H/2).
    return statistic, exp(-statistic / 2), tie_correction, mean_ranks


def analyze(rows: list[dict], parameters: dict) -> dict:
    lot = parameters.get("lot_code", DEFAULT_LOT)
    alpha = parameters.get("alpha", DEFAULT_ALPHA)
    minimum = parameters.get("min_reduction_nm", DEFAULT_MIN_REDUCTION)
    if not isinstance(lot, str) or not lot.strip():
        raise ValueError("请填写要复核的工件批号。")
    if type(alpha) not in (float, int) or not 0.001 <= alpha <= 0.2:
        raise ValueError("显著性水平须在 0.001 至 0.2 之间。")
    if type(minimum) not in (float, int) or not 0 <= minimum <= 1000:
        raise ValueError("最小有用降幅须在 0 至 1000 nm 之间。")

    lot = lot.strip()
    groups = _lot_groups(rows, lot)
    statistic, p_value, tie_correction, mean_ranks = _kruskal_wallis(groups)
    current_median = median(groups[CURRENT_CLOTH])
    results = []
    for cloth, values in groups.items():
        q1, _, q3 = quantiles(values, n=4, method="inclusive")
        group_median = median(values)
        results.append({
            "cloth": cloth,
            "n": len(values),
            "median_nm": group_median,
            "q1_nm": q1,
            "q3_nm": q3,
            "mean_rank": round(mean_ranks[cloth], 4),
            "reduction_nm": current_median - group_median,
        })
    results.sort(key=itemgetter("median_nm"))

    best_median = results[0]["median_nm"]
    reduction = current_median - best_median
    candidates = [item["cloth"] for item in results if item["median_nm"] == best_median]
    if p_value >= alpha:
        headline = "整体差异证据不足，暂缓换型"
        decision = f"整体 p 值 {p_value:.3g} 达到或高于 α={alpha:g}，建议维持现用布并补充独立工件。"
    elif reduction <= 0 or reduction < minimum:
        headline = "整体有差异，暂缓换型"
        decision = f"最大中位数降幅为 {reduction:g} nm，试产门槛要求实际降低且至少降低 {minimum:g} nm。"
    else:
        headline = "建议进入小批试产复核"
        names = "、".join(candidates)
        decision = f"优先复核 {names}，中位数较现用低 {reduction:g} nm，达到 {minimum:g} nm 试产门槛。"
    caveat = "整体检验只判断三组是否存在分布差异。试产还要用独立批次核对候选布的成对差异、划伤率和布耗。"
    return {
        "headline": headline,
        "summary": f"批次 {lot}。{decision}{caveat}",
        "metrics": [
            {"label": "独立工件数", "value": sum(map(len, groups.values())), "unit": "件"},
            {"label": "并列校正 H", "value": round(statistic, 6)},
            {"label": "整体 p 值", "value": float(f"{p_value:.6g}")},
            {"label": "并列校正系数", "value": round(tie_correction, 6)},
        ],
        "columns": [
            {"key": "cloth", "label": "抛光布"},
            {"key": "n", "label": "工件数（件）"},
            {"key": "median_nm", "label": "Ra 中位数（nm）"},
            {"key": "q1_nm", "label": "Ra 下四分位（nm）"},
            {"key": "q3_nm", "label": "Ra 上四分位（nm）"},
            {"key": "mean_rank", "label": "平均秩"},
            {"key": "reduction_nm", "label": "较现用降低（nm）"},
        ],
        "rows": results,
        "chart": {
            "label": "各布型抛光后 Ra 中位数",
            "unit": "nm",
            "points": [{"label": item["cloth"], "value": item["median_nm"]} for item in results],
        },
    }
