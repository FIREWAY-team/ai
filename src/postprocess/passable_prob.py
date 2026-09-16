"""도로 폭 계산 및 차종별 동시 통과 판정 (스펙 2-3, 2-4장, v4)."""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

from src.inference.homography import (
    combine_scales_with_error,
    depth_is_supported,
    get_footpoint,
    scale_estimates_with_history,
    vehicle_pixel_width,
    vehicle_x_span,
)
from src.inference.yolo import VehicleDetection
from src.inference.road_region import filter_road_detections
from src.schemas import MeasurementUnavailableError, ReadingCore

# verdict()의 로지스틱 기울기 — z 자체가 이미 검증된 margin_m 경계이므로
# prob는 판정과 별개로 채우는 표시용 부가값이다 (임의 확률 임계값으로
# 경계를 다시 잡지 않는다).
VERDICT_STEEPNESS = 3.0

# 장애물 폭 집계에서 "이 지점(target_y_px)을 실제로 막고 있다"고 볼 세로
# 여유 허용치. vehicle_y_span()으로 구한 차량의 실제 세로 점유 구간
# [y_min, y_max]에 target_y_px가 들어가야 포함하되, 픽셀 경계 반올림
# 오차(예: footpoint를 target_y_px로 그대로 넘길 때 1px 안팎 어긋남)를
# 흡수할 최소한의 여유만 둔다 — 카메라 높이(px)에 비례하게 잡아 해상도가
# 달라져도 같은 비율로 스케일된다.
#
# 이전에는 depth_weight(footpoint 거리 기반 지수감쇠) 임계값(0.3)으로
# "다른 깊이"를 걸렀는데, 문턱이 관대해서(카메라 높이의 ~30%) 골목 한쪽에
# 세로로 줄줄이 주차된 차들이 전부 같은 지점의 장애물로 합산되는 버그가
# 있었다. 현재는 회전 사각형의 빈 모서리까지 포함하지 않도록 해당 행 부근의
# 실제 세그멘테이션 마스크 점유 여부를 확인한다.
OBSTACLE_Y_TOLERANCE_RATIO = 0.005
OBSTACLE_Y_TOLERANCE_MIN_PX = 1.0

# 장애물 폭 집계에서 제외하는 신뢰도 하한선. A-3(REFERENCE_MIN_CONFIDENCE)와
# 같은 이유로, 특히 mAP가 낮은 클래스(트럭 0.355)는 낮은 신뢰도에서 차량이
# 아닌 것(간판 차양 등)을 차량으로 잘못 인식하는 경우가 있고, 그런 오검출은
# 세그멘테이션 마스크도 비정상적으로 커서(화면 전체에 걸친 회전사각형 등)
# 장애물 합계를 실측 벽 폭보다 크게 만들어버릴 수 있다. 이 임계값 미만은
# detected_objects에는 남기되(검출 자체는 숨기지 않음) 폭 계산에서만 뺀다.
OBSTACLE_MIN_CONFIDENCE = 0.5

# 도로 폭 계산(obstacle_width_m)에서 제외하는 클래스 (정책 확정, 스펙
# 2-1장의 미결 사항 해소). 사람은 소방차가 오면 스스로 비켜설 수 있고,
# 사람이 들고 있거나 곁에 둔 가벼운 물건도 함께 치워지므로 "차가 지나갈 수
# 없도록 막는" 장애물로 보지 않는다. 검출 자체(detected_objects)에는
# 여전히 남는다 — 폭 계산에서만 뺀다. 차량과 고정 장애물(매대·표지판 등,
# 검출 가능해지면)은 계속 포함한다.
#
# "보행자"는 2026-09-12 AI Hub 파인튜닝 클래스명(src/inference/yolo.py의
# CLASS_NAME_MAP)과 맞춘 이름 — 여기가 옛 이름("사람")으로 남아있으면 보행자가
# 조용히 장애물 폭에 합산돼버리는, 안전 방향과 반대인 회귀가 생긴다.
OBSTACLE_EXCLUDED_CLASSES = frozenset({"보행자"})


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _mask_has_pixels_at_y(mask: np.ndarray, target_y: float, tolerance: float) -> bool:
    start = max(0, math.ceil(target_y - tolerance))
    stop = min(mask.shape[0], math.floor(target_y + tolerance) + 1)
    return start < stop and bool(np.any(mask[start:stop]))


def depth_quality_flags(
    detections: Iterable[VehicleDetection],
    estimates: list[tuple[float, float, float]],
    camera_height_px: float,
    target_y_px: float | None = None,
) -> list[str]:
    """자동 탐색은 전체 장애물, 지점 지정은 해당 행의 장애물 깊이를 검사한다."""
    tolerance = max(OBSTACLE_Y_TOLERANCE_MIN_PX, camera_height_px * OBSTACLE_Y_TOLERANCE_RATIO)
    for det in detections:
        if det.vehicle_class in OBSTACLE_EXCLUDED_CLASSES or det.confidence < OBSTACLE_MIN_CONFIDENCE:
            continue
        if target_y_px is not None and not _mask_has_pixels_at_y(det.mask, target_y_px, tolerance):
            continue
        _, rect = vehicle_pixel_width(det.mask)
        if not depth_is_supported(estimates, float(get_footpoint(rect)[1])):
            return ["outside_calibration_depth"]
    return []


def obstacle_widths_m(
    detections: Iterable[VehicleDetection],
    scale_m_per_px: float,
    target_y_px: float,
    camera_height_px: float,
    min_confidence: float = OBSTACLE_MIN_CONFIDENCE,
) -> float:
    """측정 지점(target_y_px)을 실제로 막고 있는 장애물만 폭을 합산해 미터로 환산.

    화면에 찍힌 모든 검출을 원근(깊이) 구분 없이 그냥 더하면, 도로를 따라
    앞뒤로 늘어선 차량·사람까지 전부 합산돼 실제보다 훨씬 좁은 도로로
    오판한다(그 지점을 동시에 막고 있는 장애물만 그 지점의 통과폭에 영향을
    준다). target_y_px 부근에 실제 마스크 픽셀이 있을 때만 포함한다.
    회전 사각형의 빈 모서리는 차량이 없는 행까지 확장될 수 있으므로 제외한다.
    min_confidence 미만인 검출은 제외한다(정확도개선방안 A-3와 같은 이유 —
    신뢰도가 낮은 검출, 특히 mAP가 낮은 클래스는 차량이 아닌 것을 잘못
    인식했을 가능성이 있고, 그런 오검출의 마스크는 비정상적으로 커서 장애물
    합계를 실측 벽 폭보다 크게 만들 수 있다). OBSTACLE_EXCLUDED_CLASSES(사람
    등 스스로 비킬 수 있는 대상)도 제외한다.

    해당 행의 마스크 점유 검사를 통과한 차량끼리도 가로 위치(vehicle_x_span)가 겹치면
    같은 차선에 앞뒤로 붙어 있는 것이지 나란히 서서 폭을 나눠 막는 게
    아니다(2026-09-15, cctv_4 실측 검증 중 발견 — 원근 압축으로 앞차
    끝과 뒷차 시작의 세로 구간이 몇 px 겹쳐서, 뻥 뚫린 골목인데
    obstacle_width_m이 5m대로 잘못 나왔다). 가로 위치가 겹치는 차량군은
    폭을 더하지 않고 그중 가장 넓은 차 1대분만 반영하고, 가로 위치가
    겹치지 않는 차량군끼리만(다른 차선에서 동시에 좁히는 경우) 합산한다.
    """
    width, _relative_error = _obstacle_widths_at_depth(
        detections, target_y_px, camera_height_px, min_confidence,
        lambda _y: (scale_m_per_px, 0.0),
    )
    return width


def calibrated_obstacle_widths(
    detections: Iterable[VehicleDetection],
    estimates: list[tuple[float, float, float]],
    target_y_px: float,
    camera_height_px: float,
    min_confidence: float = OBSTACLE_MIN_CONFIDENCE,
) -> tuple[float, float]:
    """차량별 접지점에서 환산한 장애물 폭과 사용 스케일의 최대 상대 오차.

    target_y_px는 포함할 장애물을 고르는 행이다. 차량 전체 픽셀 폭의
    환산에는 그 차량의 접지점을 사용해야 다른 깊이의 스케일이 섞이지 않는다.
    """
    return _obstacle_widths_at_depth(
        detections, target_y_px, camera_height_px, min_confidence,
        lambda y: combine_scales_with_error(estimates, y, camera_height_px),
    )


def _obstacle_widths_at_depth(
    detections: Iterable[VehicleDetection],
    target_y_px: float,
    camera_height_px: float,
    min_confidence: float,
    scale_at_y: Callable[[float], tuple[float | None, float]],
) -> tuple[float, float]:
    tolerance_px = max(OBSTACLE_Y_TOLERANCE_MIN_PX, camera_height_px * OBSTACLE_Y_TOLERANCE_RATIO)
    spans: list[tuple[float, float, float]] = []  # (x_min, x_max, width_m)
    max_relative_error = 0.0
    for det in detections:
        if det.vehicle_class in OBSTACLE_EXCLUDED_CLASSES:
            continue
        if det.confidence < min_confidence:
            continue
        if not _mask_has_pixels_at_y(det.mask, target_y_px, tolerance_px):
            continue
        pixel_width, rect = vehicle_pixel_width(det.mask)
        _, footpoint_y = get_footpoint(rect)
        scale, error = scale_at_y(float(footpoint_y))
        if scale is None:
            raise MeasurementUnavailableError("insufficient_calibration", "장애물 접지점의 스케일을 계산할 기준 차량이 없습니다")
        max_relative_error = max(max_relative_error, error / scale if scale else 0.0)
        x_min, x_max = vehicle_x_span(rect)
        spans.append((x_min, x_max, pixel_width * scale))

    if not spans:
        return 0.0, 0.0

    # 같은 target_y_px를 점유한다고 잡힌 차량이라도, 가로 위치(차선)가 겹치면
    # 앞뒤로(같은 차선에) 겹쳐 있는 것이지 나란히 서서 폭을 나눠 막는 게
    # 아니다 — 그런 겹침 그룹은 폭을 더하지 않고 가장 넓은 차 1대분만
    # 반영한다. 가로 위치가 겹치지 않는 차량끼리만(다른 차선에서 동시에
    # 좁히는 경우) 폭을 합산한다.
    spans.sort(key=lambda s: s[0])
    total = 0.0
    cluster_end = spans[0][1]
    cluster_max_width = spans[0][2]
    for x_min, x_max, width_m in spans[1:]:
        if x_min < cluster_end:
            cluster_end = max(cluster_end, x_max)
            cluster_max_width = max(cluster_max_width, width_m)
        else:
            total += cluster_max_width
            cluster_end = x_max
            cluster_max_width = width_m
    total += cluster_max_width

    return total, max_relative_error


def compute_widths(
    detections: list[VehicleDetection],
    target_y_px: float,
    camera_height_px: float,
    wall_width_m: float,
    cctv_id: str | None = None,
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

    cctv_id — 주어지면 이 카메라에 누적된 과거 기준 차량 관측치
    (calibration_store, 정확도개선방안 A-4 확장)를 현재 프레임 것과 합쳐서
    스케일을 계산한다. CCTV는 고정 카메라라 과거 관측치도 여전히 유효하고,
    한 프레임에 기준 차량이 1~2대뿐이라 먼 지점으로 외삽할 때 생기던 큰
    오차(실측 검증 30~40%대)를 줄인다. None이면(또는 아직 누적된 게 없으면)
    기존처럼 현재 프레임만으로 계산한다 — 하위호환.
    """
    filtered = filter_road_detections(detections, cctv_id)
    if detections and not filtered:
        raise MeasurementUnavailableError("no_road_detections", "도로 영역 안에 판정할 검출이 없습니다")
    detections = filtered
    estimates = scale_estimates_with_history(detections, camera_height_px, cctv_id)
    scale, scale_error = combine_scales_with_error(estimates, target_y_px, camera_height_px)
    if scale is None:
        raise MeasurementUnavailableError("insufficient_calibration", "로컬 스케일을 계산할 기준 차량이 없습니다")

    relative_error = (scale_error / scale) if scale else 0.0
    obstacle_width_m, obstacle_relative_error = calibrated_obstacle_widths(
        detections, estimates, target_y_px, camera_height_px
    )
    # 기존 측정 행의 오차와 실제 환산에 쓴 차량별 오차 중 큰 값을 유지한다.
    calibration_error_m = max(relative_error, obstacle_relative_error) * wall_width_m
    effective_width_m = wall_width_m - obstacle_width_m
    return wall_width_m, obstacle_width_m, effective_width_m, calibration_error_m


def find_narrowest_widths(
    detections: list[VehicleDetection],
    camera_height_px: float,
    wall_width_m: float,
    cctv_id: str | None = None,
) -> tuple[float, float, float, float, float]:
    """도로의 여러 지점 중 가장 좁아지는(병목) 지점을 찾아 그 지점의
    wall_width_m·obstacle_width_m·effective_width_m·calibration_error_m을 반환한다.

    소방차는 도로의 한 지점만 지나는 게 아니라 도로 전체를 지나야 하므로,
    임의로 정한 한 지점(예: 화면의 특정 비율 지점)만 보고 판정하면 실제
    병목을 놓칠 수 있다 — 한 지점이라도 통과 못 하면 전체가 FAIL이어야
    하므로(스펙 4장 오류 비대칭성: FAIL을 PASS로 오판하는 쪽이 훨씬 위험),
    검출된 장애물들의 footpoint와 마스크 점유가 바뀌는 행을 후보로 놓고
    compute_widths()를 돌려 effective_width_m이 가장 작은(가장 좁은) 지점을
    채택한다. 반환값 마지막 원소는 채택된 target_y_px다. cctv_id는
    compute_widths()로 그대로 전달한다(누적 캘리브레이션 관측치 사용).
    """
    detections = filter_road_detections(detections, cctv_id)
    if not detections:
        raise MeasurementUnavailableError("no_road_detections", "병목 지점을 찾을 장애물 검출이 없습니다")

    candidate_ys: set[float] = set()
    tolerance_px = max(OBSTACLE_Y_TOLERANCE_MIN_PX, camera_height_px * OBSTACLE_Y_TOLERANCE_RATIO)
    for det in detections:
        _, rect = vehicle_pixel_width(det.mask)
        _, footpoint_y = get_footpoint(rect)
        if not _mask_has_pixels_at_y(det.mask, footpoint_y, tolerance_px):
            # 회전 사각형 접지점이 마스크 밖이면 실제 점유 행으로 옮겨 병목 누락 방지.
            occupied_rows = np.flatnonzero(np.any(det.mask, axis=1))
            footpoint_y = float(occupied_rows[np.argmin(np.abs(occupied_rows - footpoint_y))])
        candidate_ys.add(footpoint_y)

    # 장애물 집합은 허용폭을 포함한 점유 구간의 시작·끝에서만 바뀐다.
    # 구간 내부도 검사해 소수점 위치의 겹침과 마스크 구멍을 놓치지 않는다.
    change_rows: set[float] = set()
    for det in detections:
        if det.vehicle_class in OBSTACLE_EXCLUDED_CLASSES or det.confidence < OBSTACLE_MIN_CONFIDENCE:
            continue
        occupied = np.any(det.mask, axis=1)
        edges = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
        change_rows.update((0., float(len(occupied) - 1)))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            change_rows.add(max(0., float(start) - tolerance_px))
            change_rows.add(min(float(len(occupied) - 1), float(stop - 1) + tolerance_px))

    boundaries = sorted(change_rows)
    change_rows.update((left + right) / 2 for left, right in zip(boundaries, boundaries[1:]))
    targets = sorted(candidate_ys) + sorted(change_rows - candidate_ys)
    narrowest: tuple[float, float, float, float, float] | None = None
    for target_y in targets:
        wall_m, obstacle_m, effective_m, calib_err_m = compute_widths(
            detections, target_y, camera_height_px, wall_width_m, cctv_id=cctv_id
        )
        if narrowest is None or effective_m < narrowest[2]:
            narrowest = (wall_m, obstacle_m, effective_m, calib_err_m, target_y)
    return narrowest


def verdict(
    effective_m: float,
    vehicle_width_m: float,
    margin_m: float,
    calibration_error_m: float = 0.0,
) -> tuple[str, float]:
    """단일 차종 판정.

    calibration_error_m(이 프레임의 캘리브레이션 불일치, compute_widths 참고)이
    클수록 PASS 문턱을 더 보수적으로 넓힌다 — 좁은 골목을 비스듬히 찍은 카메라는
    기준 차량들의 스케일 추정치가 서로 크게 어긋날 수 있는데, 그런 프레임에서
    effective_width_m이 그럴듯해 보여도 실제로는 불확실한 값이다. PASS 쪽만
    넓히고 FAIL 쪽 문턱(margin_m 그대로)은 건드리지 않는다 — 불확실하다고 해서
    "위험할 수 있다"는 경고를 완화하면 안 되기 때문이다(스펙 4장 오류 비대칭성:
    FAIL을 PASS로 잘못 판정하는 쪽이 PASS를 FAIL로 잘못 판정하는 쪽보다 훨씬
    위험하다 — 정확도개선방안 D-1이 제안했던 비대칭 임계값을 calibration_error_m
    으로 구체화한 것). prob 표시값은 기존처럼 margin_m 기준 z로 계산한다(판정
    경계와는 별개의 부가값).
    """
    pass_margin_m = margin_m + calibration_error_m
    diff = effective_m - vehicle_width_m
    z = diff / margin_m
    prob = _stable_sigmoid(VERDICT_STEEPNESS * z)
    if diff >= pass_margin_m:
        status = "PASS"
    elif diff <= -margin_m:
        status = "FAIL"
    else:
        status = "UNCERTAIN"  # 소방 상황실 확인 요망, prob와 함께 표시
    return status, prob


def build_reading_verdict(
    effective_m: float,
    vehicles_json: dict[str, float],
    margin_m: float,
    calibration_error_m: float = 0.0,
) -> tuple[dict[str, str], float]:
    """차종별(pump-3.5, pump-8) 동시 판정 (스펙 2-4장).

    Reading.verdict, Reading.confidence(가장 보수적인 확률)를 만든다 —
    여러 차종 중 하나라도 확신이 낮으면 confidence 전체를 낮게 잡는다.
    """
    results: dict[str, str] = {}
    probs: list[float] = []
    for name, width in vehicles_json.items():
        status, prob = verdict(effective_m, width, margin_m, calibration_error_m)
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
    method: str = "yolov11_homography_v7",
    quality_flags: Iterable[str] = (),
) -> ReadingCore:
    """판정까지 마친 뒤 팀 공용 ReadingCore로 조립 (스펙 1장 입출력 계약)."""
    verdict_map, confidence = build_reading_verdict(
        effective_width_m, vehicles_json, margin_m, calibration_error_m
    )
    flags = set(quality_flags)
    if effective_width_m < 0:
        flags.add("negative_effective_width")
    if flags:
        verdict_map = {vehicle: "UNCERTAIN" if status == "PASS" else status
                       for vehicle, status in verdict_map.items()}
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
        quality_flags=sorted(flags),
    )
