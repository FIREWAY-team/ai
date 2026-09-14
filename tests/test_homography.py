from __future__ import annotations

import math

from src.inference.homography import (
    combine_scales,
    combine_scales_with_error,
    get_footpoint,
    is_good_reference,
    local_scale,
    local_scale_estimates,
    reject_local_outliers,
    smoothed_scale,
    vehicle_pixel_width,
)
from src.inference.yolo import VehicleDetection
from tests.helpers import make_detection, make_rect_mask, make_rotated_rect_mask


def test_vehicle_pixel_width_returns_short_side():
    mask = make_rect_mask((200, 300), x=50, y=80, w=40, h=20)
    pixel_width, rect = vehicle_pixel_width(mask)
    assert math.isclose(pixel_width, 20, abs_tol=1.0)


def test_local_scale_uses_known_vehicle_width():
    # 승용차 폭 1.8m가 픽셀 폭 90px로 찍혔다면 스케일은 0.02 m/px (신뢰도 1.0일 때)
    scale = local_scale(pixel_width=90, vehicle_class="승용차", class_conf=1.0)
    assert math.isclose(scale, 0.02, rel_tol=1e-6)


def test_local_scale_scales_with_confidence():
    full_conf = local_scale(90, "승용차", 1.0)
    half_conf = local_scale(90, "승용차", 0.5)
    assert math.isclose(half_conf, full_conf / 2, rel_tol=1e-6)


def test_get_footpoint_is_bottom_center():
    mask = make_rect_mask((200, 300), x=50, y=80, w=40, h=20)
    _, rect = vehicle_pixel_width(mask)
    fx, fy = get_footpoint(rect)
    # 사각형 하단 중앙 근처 — y는 최대값(100) 근처, x는 폭 중앙(70) 근처
    assert 95 <= fy <= 100
    assert 65 <= fx <= 75


def test_combine_scales_weights_closer_vehicle_more():
    # target_y=100 기준, 가까운 차량(y=100)과 먼 차량(y=10)이 서로 다른 스케일을 줄 때
    estimates = [(0.02, 1.0, 100.0), (0.05, 1.0, 10.0)]
    combined = combine_scales(estimates, target_y_px=100.0, camera_height_px=100.0)
    assert combined is not None
    # 가까운 차량(0.02)에 훨씬 가까운 값이어야 한다
    assert abs(combined - 0.02) < abs(combined - 0.05)


def test_combine_scales_empty_returns_none():
    assert combine_scales([], target_y_px=0.0, camera_height_px=100.0) is None


def test_combine_scales_with_error_zero_for_single_estimate():
    # 기준자 후보가 하나뿐이면 불일치를 잴 수 없으므로 오차는 0
    scale, error = combine_scales_with_error(
        [(0.02, 1.0, 100.0)], target_y_px=100.0, camera_height_px=100.0
    )
    assert scale == 0.02
    assert error == 0.0


def test_combine_scales_with_error_positive_when_estimates_disagree():
    # 같은 깊이(target_y와 동일 footpoint)인데 서로 다른 스케일을 주는 두 기준자
    # → calibration_error_m으로 노출할 불일치가 있어야 한다
    scale, error = combine_scales_with_error(
        [(0.02, 1.0, 100.0), (0.05, 1.0, 100.0)], target_y_px=100.0, camera_height_px=100.0
    )
    assert scale is not None
    assert error > 0.0


def test_combine_scales_with_error_matches_combine_scales():
    estimates = [(0.02, 1.0, 100.0), (0.05, 1.0, 10.0)]
    scale_only = combine_scales(estimates, target_y_px=100.0, camera_height_px=100.0)
    scale, _error = combine_scales_with_error(estimates, target_y_px=100.0, camera_height_px=100.0)
    assert scale == scale_only


def test_combine_scales_with_error_none_when_empty():
    scale, error = combine_scales_with_error([], target_y_px=0.0, camera_height_px=100.0)
    assert scale is None
    assert error == 0.0


def test_local_scale_estimates_from_detections():
    detections = [
        make_detection("승용차", 0.9, x=50, y=80, w=90, h=180),
    ]
    estimates = local_scale_estimates(detections)
    assert len(estimates) == 1
    scale, conf, footpoint_y = estimates[0]
    assert conf == 0.9
    assert scale > 0


def test_is_good_reference_true_for_axis_aligned_rect():
    mask = make_rect_mask((200, 300), x=50, y=80, w=40, h=90)
    _, rect = vehicle_pixel_width(mask)
    assert is_good_reference(rect) is True


def test_is_good_reference_false_for_steep_angle():
    mask = make_rotated_rect_mask((200, 300), center=(150, 100), w=40, h=90, angle_deg=45)
    _, rect = vehicle_pixel_width(mask)
    assert is_good_reference(rect) is False


def test_local_scale_estimates_excludes_low_confidence_reference():
    detections = [
        make_detection("승용차", 0.9, x=10, y=10, w=90, h=180),
        make_detection("승용차", 0.3, x=150, y=10, w=90, h=180),  # 신뢰도 하한선(0.5) 미만
    ]
    estimates = local_scale_estimates(detections)
    assert len(estimates) == 1
    assert estimates[0][1] == 0.9


def test_local_scale_estimates_prefers_good_angle_vehicles():
    good = make_detection("승용차", 0.9, x=10, y=10, w=40, h=90, shape=(200, 400))
    skewed = VehicleDetection(
        vehicle_class="승용차",
        confidence=0.95,
        bbox=(150, 10, 190, 100),
        mask=make_rotated_rect_mask((200, 400), center=(170, 55), w=40, h=90, angle_deg=45),
    )
    estimates = local_scale_estimates([good, skewed])
    # 정면 각도 차량이 하나라도 있으면 그 차량들만 채택한다
    assert len(estimates) == 1
    assert estimates[0][1] == 0.9


def test_local_scale_estimates_falls_back_when_all_skewed():
    skewed = VehicleDetection(
        vehicle_class="승용차",
        confidence=0.95,
        bbox=(150, 10, 190, 100),
        mask=make_rotated_rect_mask((200, 400), center=(170, 55), w=40, h=90, angle_deg=45),
    )
    estimates = local_scale_estimates([skewed])
    assert len(estimates) == 1


def test_smoothed_scale_filters_outlier_and_smooths():
    # median 0.02 근처에서 하나만 크게 벗어난 outlier(0.1)는 제외돼야 한다
    recent = [0.02, 0.021, 0.019, 0.1, 0.02]
    result = smoothed_scale(recent)
    assert result is not None
    assert abs(result - 0.02) < 0.01


def test_smoothed_scale_empty_returns_none():
    assert smoothed_scale([None, None]) is None


def test_reject_local_outliers_removes_deviant_same_depth_observation():
    # 비슷한 깊이(y 근처)에 모인 관측치들 사이에서 하나만 크게 벗어나면(오검출·
    # 오분류로 의심) 제외한다. 카메라 높이 1000px, window_ratio 기본 0.15면
    # 150px 이내를 "비슷한 깊이"로 본다.
    estimates = [
        (0.010, 0.9, 500.0),
        (0.0105, 0.8, 520.0),
        (0.0098, 0.85, 480.0),
        (0.05, 0.7, 510.0),  # 이웃들과 스케일이 5배 차이 — 이상치
    ]
    result = reject_local_outliers(estimates, camera_height_px=1000.0)
    scales = [s for s, _c, _y in result]
    assert 0.05 not in scales
    assert len(result) == 3


def test_reject_local_outliers_keeps_sparse_regions_untouched():
    # 비교할 이웃이 2개 미만인 깊이 구간은 이상치 판정을 안 한다 — 데이터가
    # 아직 적은 구간까지 과도하게 걸러내면 정작 필요한 관측치가 사라진다.
    estimates = [
        (0.010, 0.9, 100.0),
        (0.0105, 0.8, 110.0),
        (0.05, 0.7, 900.0),  # 근처에 비교할 이웃이 없음 — 그대로 유지
    ]
    result = reject_local_outliers(estimates, camera_height_px=1000.0)
    assert len(result) == 3


def test_reject_local_outliers_noop_under_three_estimates():
    estimates = [(0.05, 0.7, 500.0), (0.01, 0.9, 100.0)]
    result = reject_local_outliers(estimates, camera_height_px=1000.0)
    assert result == estimates
