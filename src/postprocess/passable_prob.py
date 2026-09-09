"""도로 폭 계산 및 차종별 동시 통과 판정 (스펙 2-3, 2-4장, v4)."""
from __future__ import annotations

import math
from typing import Iterable

from src.inference.homography import (
    combine_scales_with_error,
    depth_weight,
    get_footpoint,
    local_scale_estimates,
    vehicle_pixel_width,
)
from src.inference.yolo import VehicleDetection
from src.schemas import ReadingCore

# verdict()의 로지스틱 기울기 — z 자체가 이미 검증된 margin_m 경계이므로
# prob는 판정과 별개로 채우는 표시용 부가값이다 (임의 확률 임계값으로
# 경계를 다시 잡지 않는다).
VERDICT_STEEPNESS = 3.0

# 장애물 폭 집계 시 "측정 지점(target_y_px)과 다른 깊이"로 보는 가중치 하한선.
# depth_weight가 이 미만이면 그 장애물은 화면에는 찍혔어도 이 지점의 도로
# 폭에는 실질적으로 안 겹친다고 보고 제외한다 — 안 그러면 도로를 따라 앞뒤로
# 늘어선 차량까지 원근 구분 없이 다 더해져 실제보다 훨씬 좁게 오판한다.
# 초기값이며, 깊이 가중치 감쇠계수(DEPTH_DECAY_COEF)처럼 실측 데이터로
# 튜닝 대상이다.
OBSTACLE_DEPTH_WEIGHT_THRESHOLD = 0.3

# 도로 폭 계산(obstacle_width_m)에서 제외하는 클래스 (정책 확정, 스펙
# 2-1장의 미결 사항 해소). 사람은 소방차가 오면 스스로 비켜설 수 있고,
# 사람이 들고 있거나 곁에 둔 가벼운 물건도 함께 치워지므로 "차가 지나갈 수
# 없도록 막는" 장애물로 보지 않는다. 검출 자체(detected_objects)에는
# 여전히 남는다 — 폭 계산에서만 뺀다. 차량과 고정 장애물(매대·표지판 등,
# 검출 가능해지면)은 계속 포함한다.
OBSTACLE_EXCLUDED_CLASSES = frozenset({"사람"})


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def obstacle_widths_m(
    detections: Iterable[VehicleDetection],
    scale_m_per_px: float,
    target_y_px: float,
    camera_height_px: float,
    weight_threshold: float = OBSTACLE_DEPTH_WEIGHT_THRESHOLD,
) -> float:
    """측정 지점(target_y_px)과 같은 깊이에 있는 장애물만 폭을 합산해 미터로 환산.

    화면에 찍힌 모든 검출을 원근(깊이) 구분 없이 그냥 더하면, 도로를 따라
    앞뒤로 늘어선 차량·사람까지 전부 합산돼 실제보다 훨씬 좁은 도로로
    오판한다(그 지점을 동시에 막고 있는 장애물만 그 지점의 통과폭에 영향을
    준다). depth_weight가 weight_threshold 미만인 검출은 "다른 깊이"로
    보고 제외한다. OBSTACLE_EXCLUDED_CLASSES(사람 등 스스로 비킬 수 있는
    대상)도 제외한다.
    """
    total = 0.0
    for det in detections:
        if det.vehicle_class in OBSTACLE_EXCLUDED_CLASSES:
            continue
        pixel_width, rect = vehicle_pixel_width(det.mask)
        _, footpoint_y = get_footpoint(rect)
        if depth_weight(footpoint_y, target_y_px, camera_height_px) < weight_threshold:
            continue
        total += pixel_width * scale_m_per_px
    return total


def compute_widths(
    detections: list[VehicleDetection],
    target_y_px: float,
    camera_height_px: float,
    wall_width_m: float,
) -> tuple[float, float, float, float]:
    """장애물폭(obstacle_width_m)·잔여폭(effective_width_m)과 이 프레임 캘리브레이션
    추정 오차(calibration_error_m)를 계산한다 (핵심, 2-3장).

    wall_width_m — 벽~벽 실측 폭. 카카오맵/네이버지도 거리재기로 사람이 직접 잰
    값(카메라 등록 시 1회 확정, `configs/cameras.yaml`)을 그대로 받는다. SAM2
    자동 추정을 걷어냈으므로(정확도·비용 모두 지도 실측이 유리해 채택) 이
    모듈은 도로 폭 자체를 계산하지 않고, 그 실측값에서 장애물 폭만 뺀다.
    obstacle_width_m — 검출된 장애물 폭의 합. 시점마다 다르다.
    effective_width_m — wall_width_m - obstacle_width_m, 실제 판정에 쓰는 값.
    calibration_error_m은 기준 차량들의 로컬 스케일 추정치가 서로 얼마나
    불일치하는지(상대오차)를 wall_width_m에 투영한 값이다 — wall_width_m
    자체는 지도 실측이라 오차원이 아니고, 오차는 오직 장애물 폭 스케일
    추정에서만 온다.
    """
    estimates = local_scale_estimates(detections)
    scale, scale_error = combine_scales_with_error(estimates, target_y_px, camera_height_px)
    if scale is None:
        raise ValueError("로컬 스케일을 계산할 기준 차량이 없습니다")

    relative_error = (scale_error / scale) if scale else 0.0
    calibration_error_m = relative_error * wall_width_m

    obstacle_width_m = obstacle_widths_m(detections, scale, target_y_px, camera_height_px)
    effective_width_m = wall_width_m - obstacle_width_m
    return wall_width_m, obstacle_width_m, effective_width_m, calibration_error_m


def find_narrowest_widths(
    detections: list[VehicleDetection],
    camera_height_px: float,
    wall_width_m: float,
) -> tuple[float, float, float, float, float]:
    """도로의 여러 지점 중 가장 좁아지는(병목) 지점을 찾아 그 지점의
    wall_width_m·obstacle_width_m·effective_width_m·calibration_error_m을 반환한다.

    소방차는 도로의 한 지점만 지나는 게 아니라 도로 전체를 지나야 하므로,
    임의로 정한 한 지점(예: 화면의 특정 비율 지점)만 보고 판정하면 실제
    병목을 놓칠 수 있다 — 한 지점이라도 통과 못 하면 전체가 FAIL이어야
    하므로(스펙 4장 오류 비대칭성: FAIL을 PASS로 오판하는 쪽이 훨씬 위험),
    검출된 장애물들의 footpoint(깊이)를 전부 후보 지점으로 놓고 각각
    compute_widths()를 돌려 effective_width_m이 가장 작은(가장 좁은) 지점을
    채택한다. 반환값 마지막 원소는 채택된 target_y_px다.
    """
    if not detections:
        raise ValueError("병목 지점을 찾을 장애물 검출이 없습니다")

    candidate_ys: set[float] = set()
    for det in detections:
        _, rect = vehicle_pixel_width(det.mask)
        _, footpoint_y = get_footpoint(rect)
        candidate_ys.add(footpoint_y)

    narrowest: tuple[float, float, float, float, float] | None = None
    for target_y in candidate_ys:
        wall_m, obstacle_m, effective_m, calib_err_m = compute_widths(
            detections, target_y, camera_height_px, wall_width_m
        )
        if narrowest is None or effective_m < narrowest[2]:
            narrowest = (wall_m, obstacle_m, effective_m, calib_err_m, target_y)
    return narrowest


def verdict(effective_m: float, vehicle_width_m: float, margin_m: float) -> tuple[str, float]:
    """단일 차종 판정. 판정 경계(z >= 1 / z <= -1)는 검증된 margin_m 그대로 쓰고,
    prob는 판정과 별도로 계산해 채우는 부가값(소방 상황실 표시용)일 뿐이다.
    """
    z = (effective_m - vehicle_width_m) / margin_m
    prob = _stable_sigmoid(VERDICT_STEEPNESS * z)
    if z >= 1:
        status = "PASS"
    elif z <= -1:
        status = "FAIL"
    else:
        status = "UNCERTAIN"  # 소방 상황실 확인 요망, prob와 함께 표시
    return status, prob


def build_reading_verdict(
    effective_m: float, vehicles_json: dict[str, float], margin_m: float
) -> tuple[dict[str, str], float]:
    """차종별(pump-3.5, pump-8) 동시 판정 (스펙 2-4장).

    Reading.verdict, Reading.confidence(가장 보수적인 확률)를 만든다 —
    여러 차종 중 하나라도 확신이 낮으면 confidence 전체를 낮게 잡는다.
    """
    results: dict[str, str] = {}
    probs: list[float] = []
    for name, width in vehicles_json.items():
        status, prob = verdict(effective_m, width, margin_m)
        results[name] = status
        probs.append(prob)
    return results, min(probs)


def vote_status(recent_statuses: list[str], min_pass_count: int = 3) -> str:
    """최근 N프레임의 verdict() 결과를 다수결로 최종 확정한다 (정확도개선방안 C-3).

    순간적 검출 노이즈로 인한 오판을 줄인다. PASS가 min_pass_count 이상이면
    PASS, FAIL이 과반이면 FAIL, 그 외는 UNCERTAIN(애매하면 보수적으로 확인 요망).
    """
    if not recent_statuses:
        return "UNCERTAIN"
    pass_count = recent_statuses.count("PASS")
    fail_count = recent_statuses.count("FAIL")
    if pass_count >= min_pass_count:
        return "PASS"
    if fail_count > len(recent_statuses) / 2:
        return "FAIL"
    return "UNCERTAIN"


def build_reading_core(
    wall_width_m: float,
    obstacle_width_m: float,
    effective_width_m: float,
    detections: list[VehicleDetection],
    vehicles_json: dict[str, float],
    margin_m: float,
    calibration_error_m: float = 0.0,
    method: str = "yolov11_homography_v1",
) -> ReadingCore:
    """판정까지 마친 뒤 팀 공용 ReadingCore로 조립 (스펙 1장 입출력 계약)."""
    verdict_map, confidence = build_reading_verdict(effective_width_m, vehicles_json, margin_m)
    return ReadingCore(
        wall_width_m=wall_width_m,
        obstacle_width_m=obstacle_width_m,
        effective_width_m=effective_width_m,
        detected_objects=[
            {
                "vehicle_class": det.vehicle_class,
                "confidence": det.confidence,
                "bbox": det.bbox,
            }
            for det in detections
        ],
        verdict=verdict_map,
        confidence=confidence,
        calibration_error_m=calibration_error_m,
        method=method,
    )
