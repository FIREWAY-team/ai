"""로컬 스케일 캘리브레이션 — 차량 기반 (스펙 2-2장).

이름은 팀 스펙(homography.py)을 유지하지만, 방문 답사 없이 화면 속 차량을
임시 자로 쓰는 로컬 스케일 방식으로 구현한다.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable

import cv2
import numpy as np

from goldenlane_vehicle_specs import VEHICLE_WIDTH_M
from src.inference.yolo import VehicleDetection

# 깊이 가중치 감쇠 계수 — 초기값, 9/10 1차 실측 방문 데이터로 확정됨 (스펙 2-2 한계 참고)
DEPTH_DECAY_COEF = 4.0

# 기준 차량(자로 쓸 차량) 채택 각도 허용치 — 카메라 정면 기준 ±15도
# (정확도개선방안 A-2). 벗어나면 minAreaRect의 "짧은 변"이 투영 왜곡으로 커진다.
REFERENCE_ANGLE_THRESHOLD_DEG = 15.0

# 캘리브레이션 기준 차량 후보의 신뢰도 하한선 (정확도개선방안 A-3).
# 판정 대상 차종 검출에는 영향 없음 — "자로 쓸 차량" 선정에만 적용한다.
REFERENCE_MIN_CONFIDENCE = 0.5


def vehicle_pixel_width(mask: np.ndarray) -> tuple[float, tuple]:
    """세그멘테이션 마스크에서 회전사각형 폭(짧은 변)과 rect를 구한다."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("차량 마스크에서 윤곽선을 찾지 못했습니다")
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    pixel_width = min(rect[1])
    return pixel_width, rect


def local_scale(pixel_width: float, vehicle_class: str, class_conf: float) -> float:
    """알려진 차종 폭으로 그 지점의 로컬 스케일(m/px)을 구한다."""
    return VEHICLE_WIDTH_M[vehicle_class] / pixel_width * class_conf


def get_footpoint(rect: tuple) -> tuple[float, float]:
    """지면 접촉점(footpoint) — 회전사각형 하단 두 꼭짓점의 중점.

    bbox 중심/centroid는 차량 높이만큼 깊이(원근)를 왜곡시키므로 쓰지 않는다 —
    화면 y좌표와 실제 깊이의 상관관계는 지면 접촉점에서만 성립한다.
    """
    box = cv2.boxPoints(rect)
    bottom_two = sorted(box, key=lambda p: p[1], reverse=True)[:2]
    return (bottom_two[0][0] + bottom_two[1][0]) / 2, (bottom_two[0][1] + bottom_two[1][1]) / 2


def vehicle_y_span(rect: tuple) -> tuple[float, float]:
    """회전사각형이 화면 세로축으로 차지하는 구간(y_min, y_max) — 차량이 실제로
    "그 깊이 구간"을 점유한다고 볼 수 있는 범위다 (정확도개선방안 버그 수정,
    2026-09-11: 골목 한쪽에 세로로 줄줄이 주차된 차들을 footpoint 거리 기반
    감쇠 가중치만으로 걸러내면, 감쇠 문턱값이 관대해서 서로 다른 지점에 서
    있는 차들이 전부 같은 지점의 장애물로 합산돼 버린다 — 실제 차량 폭보다
    도로가 좁게 나오는 정도가 아니라 벽 실측폭보다 장애물 합이 더 커지는
    수준의 오류였다. 이 함수로 차량의 실제 세로 점유 구간을 구해서, 측정
    지점(target_y_px)이 그 구간 안에 들 때만 "그 지점을 막고 있다"고
    판정한다(obstacle_widths_m 참고).
    """
    box = cv2.boxPoints(rect)
    ys = box[:, 1]
    return float(ys.min()), float(ys.max())


def is_good_reference(rect: tuple, angle_threshold_deg: float = REFERENCE_ANGLE_THRESHOLD_DEG) -> bool:
    """카메라 광축에 대해 비스듬히 주차된 차량은 기준자로 부적합하다 (정확도개선방안 A-2).

    minAreaRect의 회전각이 크면 "짧은 변"이 투영 왜곡으로 실제 차폭보다
    커진다. 각도는 0-90도로 정규화해 카메라 정면(0도/90도) 기준 편차를 잰다.
    """
    angle = rect[2] % 90
    return min(angle, 90 - angle) <= angle_threshold_deg


def local_scale_estimates(
    detections: Iterable[VehicleDetection],
    min_confidence: float = REFERENCE_MIN_CONFIDENCE,
) -> list[tuple[float, float, float]]:
    """VehicleDetection 목록 → combine_scales가 받는 (scale, conf, footpoint_y) 리스트.

    기준 차량(자로 쓸 차량) 선정에만 두 가지 필터를 적용한다 (판정 대상
    차종 검출 자체에는 영향 없음, 정확도개선방안 A-2/A-3):
    - VEHICLE_WIDTH_M에 알려진 폭이 없는 클래스(사람 등 장애물 전용 검출,
      스펙 2-1장)는 애초에 자로 쓸 수 없으므로 후보에서 제외한다.
    - 신뢰도가 min_confidence 미만인 검출은 후보에서 제외한다.
    - 정면 각도(±angle_threshold_deg) 차량이 하나라도 있으면 그 차량들만 쓴다.
      전부 비스듬하면(회전 보정 계수 미확정, 실측 데이터 필요) 어쩔 수 없이
      전체를 그대로 쓴다 — 원근 왜곡 위험은 combine_scales의 깊이 가중치가 일부 보완한다.
    """
    candidates = [
        det
        for det in detections
        if det.vehicle_class in VEHICLE_WIDTH_M and det.confidence >= min_confidence
    ]

    rects = {id(det): vehicle_pixel_width(det.mask) for det in candidates}
    good_angle = [det for det in candidates if is_good_reference(rects[id(det)][1])]
    reference_pool = good_angle if good_angle else candidates

    estimates = []
    for det in reference_pool:
        pixel_width, rect = rects[id(det)]
        scale = local_scale(pixel_width, det.vehicle_class, det.confidence)
        _, footpoint_y = get_footpoint(rect)
        estimates.append((scale, det.confidence, footpoint_y))
    return estimates


DEPTH_OUTLIER_WINDOW_RATIO = 0.15
DEPTH_OUTLIER_MAX_DEVIATION_RATIO = 0.5


def reject_local_outliers(
    estimates: list[tuple[float, float, float]],
    camera_height_px: float,
    window_ratio: float = DEPTH_OUTLIER_WINDOW_RATIO,
    max_deviation_ratio: float = DEPTH_OUTLIER_MAX_DEVIATION_RATIO,
) -> list[tuple[float, float, float]]:
    """여러 프레임에 걸쳐 누적된 기준 차량 관측치에서, 비슷한 깊이(footpoint_y)
    구간의 다른 관측치들과 스케일이 크게 어긋나는 것을 제외한다.

    smoothed_scale()의 median 기반 이상치 제거와 같은 원리를 "시간"이 아니라
    "깊이" 축에 적용한 것 — 오검출·오분류 하나가 calibration_store에 영구
    누적되면(정확도개선방안 A-4 확장) 그 이후 판정에 계속 영향을 주므로, 쌓인
    관측치끼리 서로 검증하게 한다. 비교할 이웃이 2개 미만인 구간(데이터가
    아직 적은 깊이대)은 과도하게 걸러내지 않도록 그대로 통과시킨다.
    """
    if len(estimates) < 3:
        return list(estimates)

    window_px = camera_height_px * window_ratio
    kept: list[tuple[float, float, float]] = []
    for i, (scale, conf, y) in enumerate(estimates):
        neighbors = [s for j, (s, _c, ny) in enumerate(estimates) if j != i and abs(ny - y) <= window_px]
        if len(neighbors) < 2:
            kept.append((scale, conf, y))
            continue
        med = statistics.median(neighbors)
        if med == 0 or abs(scale - med) / med <= max_deviation_ratio:
            kept.append((scale, conf, y))
    return kept


def smoothed_scale(recent_scales: list[float | None], alpha: float = 0.3) -> float | None:
    """같은 카메라의 최근 N프레임 스케일 추정치를 EMA로 스무딩한다 (정확도개선방안 A-4).

    median에서 30% 이상 벗어난 프레임(순간적 오검출·부분 가림)은 outlier로
    보고 제외한 뒤, 오래된 것부터 지수이동평균한다. recent_scales는 오래된
    프레임이 먼저 오는 순서로 전달한다.
    """
    valid = [s for s in recent_scales if s is not None]
    if not valid:
        return None
    med = statistics.median(valid)
    filtered = [s for s in valid if med == 0 or abs(s - med) / med < 0.3]
    if not filtered:
        return med
    ema = filtered[0]
    for s in filtered[1:]:
        ema = alpha * s + (1 - alpha) * ema
    return ema


def combine_scales(
    estimates: list[tuple[float, float, float]],
    target_y_px: float,
    camera_height_px: float,
) -> float | None:
    """
    estimates: [(scale, class_conf, vehicle_footpoint_y_px), ...]
    target_y_px: 측정 대상(장애물)의 footpoint 화면 y좌표 — 기준 차량과
        동일 정의(get_footpoint)로 구한다.

    한계 — 재려는 지점이 기준 차량과 비슷한 깊이(카메라로부터 거리)에 있을 때
    가장 정확하다. 먼 차량의 기여도는 깊이 가중치로 낮춰 원근 오차를 보완한다.
    """
    scale, _error = combine_scales_with_error(estimates, target_y_px, camera_height_px)
    return scale


def depth_weight(
    vehicle_y_px: float,
    target_y_px: float,
    camera_height_px: float,
    decay_coef: float = DEPTH_DECAY_COEF,
) -> float:
    """측정 대상 지점(target_y_px)과 화면상 세로 거리가 멀수록(원근상 깊이
    차이가 클수록) 기여도를 지수적으로 낮춘다. combine_scales_with_error(캘리브레이션)와
    obstacle_widths_m(장애물 폭 집계) 양쪽에서 같은 개념 — "그 지점과 같은 깊이에
    있는 것만 그 지점에 유의미하게 기여한다"는 원리를 공유한다.
    """
    dist = abs(vehicle_y_px - target_y_px) / camera_height_px
    return math.exp(-decay_coef * dist)


def combine_scales_with_error(
    estimates: list[tuple[float, float, float]],
    target_y_px: float,
    camera_height_px: float,
) -> tuple[float | None, float]:
    """combine_scales와 동일한 가중평균에 더해, 기준자 후보 간 불일치를
    `Reading.calibration_error_m`(스펙 1장)으로 노출한다.

    기준자가 여럿인데 서로 다른 스케일을 준다면 이 프레임의 캘리브레이션은
    신뢰도가 낮다는 뜻이므로, 가중 표준편차를 오차로 쓴다. 기준자가 하나뿐이면
    불일치를 잴 수 없으므로 오차는 0으로 둔다(과소평가 위험은 있으나, 후보가
    하나뿐인 프레임 자체가 이미 신뢰도 하한선(min_confidence)과 각도 필터를
    통과한 것이므로 최선의 근사치다).
    """
    weights = [(s, c * depth_weight(y, target_y_px, camera_height_px)) for s, c, y in estimates]
    total_w = sum(w for _, w in weights)
    if not total_w:
        return None, 0.0

    combined = sum(s * w for s, w in weights) / total_w
    if len(weights) < 2:
        return combined, 0.0

    variance = sum(w * (s - combined) ** 2 for s, w in weights) / total_w
    return combined, math.sqrt(variance)
