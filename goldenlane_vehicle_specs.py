"""차종별 폭 규격과 판정 여유폭(margin) 산정 (스펙 1, 2-4장, v4)."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

# 로컬 스케일 캘리브레이션(2-2장)에서 "자"로 쓰는 일반 차종 폭 — YOLO11-seg 검출
# 클래스명(src/inference/yolo.py의 CLASS_NAME_MAP)과 키를 맞춘다.
VEHICLE_WIDTH_M: dict[str, float] = {
    "승용차": 1.8,
    "버스": 2.5,
    "오토바이": 0.8,
    "트럭": 2.5,
}

VEHICLES_JSON_PATH = Path(__file__).parent / "configs" / "vehicles.json"


def load_vehicles_json(path: Path = VEHICLES_JSON_PATH) -> dict[str, float]:
    """`verdict()` 대상 소방 차종(pump-3.5, pump-8) 폭(m) — 윤종호 산출물,
    `configs/vehicles.json`이 최종 확정본. 여기서는 캘리브레이션용 일반
    차종(VEHICLE_WIDTH_M)과 섞이지 않도록 분리해서 로드한다.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# 소방 실무 최소 여유 기준 — 측면 여유 25cm (goldenlane-vehicle-specs.md 2장)
MIN_MARGIN_M = 0.25


def resolve_margin_m(mean_error_m: float | None = None, std_error_m: float = 0.0) -> float:
    """margin_m 확정 — 실측 오차율(평균 + 2×표준편차, 95% 신뢰수준 근사)과
    소방 실무 최소 여유(25cm) 중 큰 값 (정확도개선방안 C-2).

    측정 정밀도가 좋아져도 margin은 0.25m 밑으로 내려가지 않는다 — 이건 측정
    오차가 아니라 현장 실무 기준이기 때문 (스펙 2-4장).

    mean_error_m이 없으면(아직 실측 전) 최소 기준값만 반환한다.
    """
    if mean_error_m is None:
        return MIN_MARGIN_M
    measured_margin = mean_error_m + 2 * std_error_m
    # 최소값을 첫 번째 인자로 둬 NaN 실측값도 최소 안전 여유보다 우선하지
    # 못하게 한다. ``max(float('nan'), MIN_MARGIN_M)``는 NaN을 반환한다.
    return max(MIN_MARGIN_M, measured_margin)


def resolve_margin_m_from_samples(field_error_rates_m: list[float]) -> float:
    """5장 실측 5곳에서 나온 |측정값 - 실제값| 오차 샘플로부터 margin_m을 계산한다
    (정확도개선방안 C-2). 표본이 1개뿐이면 표준편차는 0으로 취급한다.

    5곳뿐이라 표준편차 추정이 불안정할 수 있음 — 지점을 늘리면 신뢰도가 오른다
    (원본 문서 5장 "여건이 되면 1곳은 줄자로 대조").
    """
    if not field_error_rates_m:
        return MIN_MARGIN_M
    mean_err = statistics.mean(field_error_rates_m)
    std_err = statistics.stdev(field_error_rates_m) if len(field_error_rates_m) > 1 else 0.0
    return resolve_margin_m(mean_err, std_err)
