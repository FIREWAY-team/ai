"""단일 카메라 데모: 정지/이동 물체 판별 (스펙 6장, 범위 한정 기능).

메인 passable_prob.py/입력 어댑터 구조를 건드리지 않는 별도 데모 스크립트다.
최종 출력은 다른 카메라와 동일하게 ReadingCore 형태를 유지하되, method
필드만 구분해 표시한다. 라우팅 모듈 쪽 인터페이스 계약은 변경 없음.

주의 — 관찰 윈도우가 짧으면 신호 대기 차량을 정지로 오분류할 수 있다.
데모용 카메라는 신호등 없는 골목으로 선정해서 이 문제를 피한다.
"""
from __future__ import annotations

import math

import numpy as np

from goldenlane_vehicle_specs import load_vehicles_json, resolve_margin_m
from src.inference import yolo
from src.inference.road_region import filter_road_detections
from src.inference.homography import combine_scales_with_error, depth_is_supported, get_footpoint, scale_estimates_with_history, vehicle_pixel_width
from src.postprocess.passable_prob import (
    OBSTACLE_EXCLUDED_CLASSES, OBSTACLE_MIN_CONFIDENCE,
    build_reading_core, calibrated_obstacle_widths, depth_quality_flags, find_narrowest_widths,
)
from src.schemas import ReadingCore

MOTION_METHOD = "yolov11_homography_v5_motion_aware"
SPEED_THRESHOLD_M_PER_SEC = 0.3
IOU_MATCH_THRESHOLD = 0.3


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def track_vehicles(
    frames_detections: list[list[yolo.VehicleDetection]],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> tuple[dict[int, list[tuple[float, float, int]]], dict[int, yolo.VehicleDetection]]:
    """경량 IoU 트래커 — 새 모델 학습 없이 프레임 간 bbox를 매칭한다.

    반환값: (track_id별 footpoint 궤적, 마지막 프레임에서 track_id → 검출 매핑).
    """
    tracks: dict[int, list[tuple[float, float, int]]] = {}
    active: dict[int, yolo.VehicleDetection] = {}
    next_id = 0
    last_frame_assignment: dict[int, yolo.VehicleDetection] = {}

    for frame_idx, detections in enumerate(frames_detections):
        matched_ids: set[int] = set()
        last_frame_assignment = {}
        for det in detections:
            best_id, best_iou = None, iou_threshold
            for track_id, last_det in active.items():
                if track_id in matched_ids:
                    continue
                iou = _iou(det.bbox, last_det.bbox)
                if iou > best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is None:
                best_id = next_id
                next_id += 1
                tracks[best_id] = []
            _, rect = vehicle_pixel_width(det.mask)
            fx, fy = get_footpoint(rect)
            tracks[best_id].append((fx, fy, frame_idx))
            active[best_id] = det
            matched_ids.add(best_id)
            last_frame_assignment[best_id] = det

    return tracks, last_frame_assignment


def classify_motion(
    tracked_vehicles: dict[int, list[tuple[float, float, int]]],
    scale_m_per_px: float,
    frame_interval_sec: float = 1.0,
    speed_threshold_m_per_sec: float = SPEED_THRESHOLD_M_PER_SEC,
) -> dict[int, str]:
    """같은 카메라(고정 시점)에서 짧은 시간차로 받은 프레임들에 걸쳐
    위치 변화량을 속도(m/s)로 환산해 정지/이동을 판별한다.
    frame_idx는 같은 간격으로 입력된 전체 프레임 기준이며 검출 누락도 포함한다.
    """
    if not math.isfinite(frame_interval_sec) or frame_interval_sec <= 0:
        raise ValueError("프레임 간격은 유한한 양수여야 합니다")
    results: dict[int, str] = {}
    for track_id, positions in tracked_vehicles.items():
        if len(positions) < 2:
            results[track_id] = "UNKNOWN"  # 한 프레임에만 나타남 — 판단 보류
            continue
        if any(current[2] <= previous[2] for previous, current in zip(positions, positions[1:])):
            results[track_id] = "UNKNOWN"
            continue
        (x1, y1, first_frame), (x2, y2, last_frame) = positions[0], positions[-1]
        displacement_m = math.hypot(x2 - x1, y2 - y1) * scale_m_per_px
        # 검출이 두 번이어도 9프레임 간격이면 경과 시간은 9프레임분이다.
        elapsed_sec = (last_frame - first_frame) * frame_interval_sec
        speed = displacement_m / elapsed_sec
        results[track_id] = "MOVING" if speed >= speed_threshold_m_per_sec else "STATIONARY"
    return results


def run_motion_aware_demo(
    frames: list[np.ndarray],
    wall_width_m: float,
    target_y_px: float | None,
    camera_height_px: float,
    vehicles_json: dict[str, float] | None = None,
    frame_interval_sec: float = 1.0,
    margin_m: float | None = None,
    cctv_id: str | None = None,
) -> ReadingCore:
    """정지/이동 판별까지 포함한 단일 카메라 데모 흐름.

    wall_width_m은 카메라 등록 시 지도 실측으로 확정한 값(`configs/cameras.yaml`).
    STATIONARY와 UNKNOWN 차량은 장애물 후보로 유지한다. MOVING으로
    확인된 차량만 제외한다. 보행자·검출 신뢰도 필터는 폭 계산에서 적용한다.

    target_y_px — 명시하면(기존 동작) 그 지점만 계산한다. None이면(기본,
    2026-09-15부터) find_narrowest_widths로 장애물 후보들 중 병목 지점을
    자동으로 찾는다 — 이미지 경로(pipeline.process_frame)는 이미 이렇게
    바뀌었는데 영상 경로는 고정 지점에 남아있어서, 병목탐색으로 잡히는
    장애물을 놓치는 카메라가 있었다(실측 검증 중 발견). 이동 판별용
    스케일(motion_scale)은 병목탐색과 무관하게 대략적인 값이면 충분해서
    (STATIONARY/MOVING을 가르는 속도 임계값 비교용) target_y_px가 없으면
    화면 70% 지점 기준으로 한 번만 계산한다.

    cctv_id — 이동 속도 판별과 폭 계산 모두 누적 관측치를 사용한다.
    마지막 프레임에 기준 차량이 없어도 같은 프레임 높이의 과거 관측치로
    계산 가능하다. find_narrowest_widths()에도 그대로 전달돼 이 카메라에
    누적된 과거 기준 차량 관측치(calibration_store)를 현재 프레임 것과
    합쳐 스케일을 계산한다. 이미지 경로(pipeline.process_frame)는 이미
    cctv_id를 받는데 영상 경로는 빠져 있었다(2026-09-15 발견 — 영상
    카메라들도 scripts/accumulate_calibration.py로 관측치를 쌓아놨지만
    실제로는 안 쓰이고 있었다).
    """
    if len(frames) < 2:
        raise ValueError("최소 2개 이상의 프레임이 필요합니다")

    frames_detections = [yolo.detect_vehicles(f) for f in frames]
    last_detections = frames_detections[-1]
    frames_detections = [
        filter_road_detections(dets, cctv_id, frame.shape)
        for dets, frame in zip(frames_detections, frames)
    ]
    if last_detections and not frames_detections[-1]:
        raise ValueError("도로 영역 안에 판정할 검출이 없습니다")

    motion_target_y = target_y_px if target_y_px is not None else camera_height_px * 0.7
    estimates = scale_estimates_with_history(frames_detections[-1], camera_height_px, cctv_id)
    motion_scale, scale_error = combine_scales_with_error(estimates, motion_target_y, camera_height_px)
    if motion_scale is None:
        raise ValueError("기준 차량을 찾지 못해 스케일을 계산할 수 없습니다")

    tracks, last_frame_assignment = track_vehicles(frames_detections)
    motion = classify_motion(tracks, motion_scale, frame_interval_sec)
    quality_flags = []
    # 외삽한 이동 스케일로 차량을 제외했다면 빈 도로 PASS로 확정하지 않는다.
    moving_obstacles = [det for tid, det in last_frame_assignment.items()
                        if motion.get(tid) == "MOVING" and det.confidence >= OBSTACLE_MIN_CONFIDENCE
                        and det.vehicle_class not in OBSTACLE_EXCLUDED_CLASSES]
    if moving_obstacles and not depth_is_supported(estimates, motion_target_y):
        quality_flags.append("motion_scale_outside_calibration_depth")
    quality_flags.extend(depth_quality_flags(frames_detections[-1], estimates, camera_height_px, target_y_px))

    # 관측 부족은 이동 증거가 아니다. 처음 등장한 차량도 현재 통로를 막을 수 있다.
    blocking_detections = [
        det for tid, det in last_frame_assignment.items() if motion.get(tid) != "MOVING"
    ]

    if target_y_px is not None:
        obstacle_width_m, obstacle_relative_error = calibrated_obstacle_widths(
            blocking_detections, estimates, target_y_px, camera_height_px
        )
        effective_width_m = wall_width_m - obstacle_width_m
        calibration_error_m = max(scale_error / motion_scale, obstacle_relative_error) * wall_width_m
    elif not blocking_detections:
        obstacle_width_m = 0.0
        effective_width_m = wall_width_m
        calibration_error_m = 0.0
    else:
        blocking_estimates = scale_estimates_with_history(blocking_detections, camera_height_px, cctv_id)
        quality_flags.extend(depth_quality_flags(blocking_detections, blocking_estimates, camera_height_px))
        _wall, obstacle_width_m, effective_width_m, calibration_error_m, _bottleneck_y = (
            find_narrowest_widths(
                blocking_detections, camera_height_px, wall_width_m, cctv_id=cctv_id
            )
        )

    resolved_vehicles = vehicles_json if vehicles_json is not None else load_vehicles_json()
    resolved_margin = margin_m if margin_m is not None else resolve_margin_m()
    return build_reading_core(
        wall_width_m,
        obstacle_width_m,
        effective_width_m,
        last_detections,
        resolved_vehicles,
        resolved_margin,
        calibration_error_m=calibration_error_m,
        method=MOTION_METHOD,
        quality_flags=quality_flags,
    )
