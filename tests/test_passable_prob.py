from __future__ import annotations

import math

from goldenlane_vehicle_specs import (
    MIN_MARGIN_M,
    resolve_margin_m,
    resolve_margin_m_from_samples,
)
from src.postprocess.passable_prob import (
    build_reading_core,
    build_reading_verdict,
    compute_widths,
    find_narrowest_widths,
    obstacle_widths_m,
    verdict,
    vote_status,
)
from tests.helpers import make_detection

VEHICLES_JSON = {"pump-3.5": 2.3, "pump-8": 2.5}


def test_resolve_margin_m_defaults_to_min():
    assert resolve_margin_m() == MIN_MARGIN_M


def test_resolve_margin_m_never_goes_below_min():
    # 실측 오차가 최소 기준보다 작아도 0.25m 밑으로 내려가지 않는다
    assert resolve_margin_m(mean_error_m=0.05, std_error_m=0.02) == MIN_MARGIN_M


def test_resolve_margin_m_nan_measurement_falls_back_to_minimum():
    # Python의 max(nan, minimum)는 nan을 반환하므로, 최소 안전 여유를 첫
    # 번째 인자로 두지 않으면 NaN 실측 입력이 그대로 전파될 수 있다.
    assert resolve_margin_m(mean_error_m=math.nan) == MIN_MARGIN_M


def test_resolve_margin_m_uses_measured_when_larger():
    # 평균 + 2*표준편차(95% 신뢰수준 근사) — 정확도개선방안 C-2
    margin = resolve_margin_m(mean_error_m=0.3, std_error_m=0.1)
    assert math.isclose(margin, 0.5)


def test_resolve_margin_m_from_samples_matches_manual_calc():
    margin = resolve_margin_m_from_samples([0.1, 0.2, 0.3])
    mean = 0.2
    stdev = math.sqrt(((0.1 - mean) ** 2 + (0.2 - mean) ** 2 + (0.3 - mean) ** 2) / 2)
    expected = max(MIN_MARGIN_M, mean + 2 * stdev)
    assert math.isclose(margin, expected)


def test_resolve_margin_m_from_samples_empty_returns_min():
    assert resolve_margin_m_from_samples([]) == MIN_MARGIN_M


def test_verdict_pass_when_z_at_least_one():
    need = VEHICLES_JSON["pump-3.5"]
    margin_m = 0.25
    effective_m = need + margin_m  # z == 1
    status, prob = verdict(effective_m, need, margin_m)
    assert status == "PASS"
    assert prob > 0.5


def test_verdict_fail_when_z_at_most_minus_one():
    need = VEHICLES_JSON["pump-3.5"]
    margin_m = 0.25
    effective_m = need - margin_m  # z == -1
    status, prob = verdict(effective_m, need, margin_m)
    assert status == "FAIL"
    assert prob < 0.5


def test_verdict_uncertain_between_boundaries():
    need = VEHICLES_JSON["pump-3.5"]
    margin_m = 0.25
    effective_m = need  # z == 0
    status, _prob = verdict(effective_m, need, margin_m)
    assert status == "UNCERTAIN"


def test_verdict_extreme_effective_widths_are_overflow_safe():
    for effective_m, expected_status in [(-100.0, "FAIL"), (100.0, "PASS")]:
        status, prob = verdict(effective_m, vehicle_width_m=2.3, margin_m=0.25)
        assert status == expected_status
        assert 0.0 <= prob <= 1.0


def test_build_reading_verdict_judges_all_vehicle_types_at_once():
    # 잔여폭 2.6m: pump-3.5(2.3m, margin 0.25)는 z=1.2로 PASS, pump-8(2.5m)은 z=0.4로 UNCERTAIN
    effective_m, margin_m = 2.6, 0.25
    results, confidence = build_reading_verdict(effective_m, VEHICLES_JSON, margin_m)
    assert results["pump-3.5"] == "PASS"
    assert results["pump-8"] == "UNCERTAIN"
    # confidence는 차종별 확률 중 가장 보수적인(작은) 값이어야 한다
    _status_35, prob_35 = verdict(effective_m, VEHICLES_JSON["pump-3.5"], margin_m)
    _status_8, prob_8 = verdict(effective_m, VEHICLES_JSON["pump-8"], margin_m)
    assert math.isclose(confidence, min(prob_35, prob_8))


def test_obstacle_widths_m_sums_pixel_widths_scaled():
    # 두 detection 모두 footpoint y가 190으로 같다(같은 깊이) — target_y_px를
    # 그 지점으로 잡으면 둘 다 "그 지점의 도로 폭"에 포함된다.
    detections = [
        make_detection("승용차", 0.9, x=10, y=10, w=90, h=180),
        make_detection("승용차", 0.9, x=150, y=10, w=90, h=180),
    ]
    total = obstacle_widths_m(
        detections, scale_m_per_px=0.02, target_y_px=190.0, camera_height_px=200.0
    )
    # cv2.minAreaRect는 픽셀 경계 관례상 채운 폭(90px)보다 1px 적게 반환한다
    assert math.isclose(total, 89 * 0.02 * 2, rel_tol=1e-6)


def test_obstacle_widths_m_excludes_detections_at_a_different_depth():
    # target_y_px(190)와 화면상 세로로 멀리 떨어진(깊이가 다른) 장애물은
    # 화면에 찍혀 있어도 이 지점의 도로 폭 계산에서 빠져야 한다.
    same_depth = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(400, 300))  # footpoint=190
    far_depth = make_detection("승용차", 0.9, x=150, y=350, w=90, h=30, shape=(400, 300))  # footpoint=380

    total_both = obstacle_widths_m(
        [same_depth, far_depth], scale_m_per_px=0.02, target_y_px=190.0, camera_height_px=400.0
    )
    total_same_only = obstacle_widths_m(
        [same_depth], scale_m_per_px=0.02, target_y_px=190.0, camera_height_px=400.0
    )
    assert math.isclose(total_both, total_same_only)


def test_obstacle_widths_m_excludes_people():
    # 사람은 소방차가 오면 스스로 비켜설 수 있으므로 도로 폭 계산에서
    # 제외한다(정책 확정) — 검출은 되지만 폭 합산에는 안 들어간다.
    car = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180)  # footpoint=190
    person = make_detection("사람", 0.8, x=150, y=10, w=40, h=180)  # 같은 깊이(footpoint=190)

    total_car_only = obstacle_widths_m(
        [car], scale_m_per_px=0.02, target_y_px=190.0, camera_height_px=200.0
    )
    total_with_person = obstacle_widths_m(
        [car, person], scale_m_per_px=0.02, target_y_px=190.0, camera_height_px=200.0
    )
    assert math.isclose(total_car_only, total_with_person)


def test_compute_widths_effective_is_less_than_wall():
    detections = [
        make_detection("승용차", 0.9, x=10, y=10, w=90, h=180),
    ]
    # target_y_px를 detection의 footpoint(190)와 같은 깊이로 잡아야 장애물로 반영된다.
    wall_m, obstacle_m, effective_m, calib_err_m = compute_widths(
        detections, target_y_px=190.0, camera_height_px=200.0, wall_width_m=4.2
    )
    assert wall_m == 4.2
    assert obstacle_m > 0.0
    assert effective_m == wall_m - obstacle_m
    assert effective_m < wall_m
    # 기준자 후보가 하나뿐이면 불일치를 잴 수 없으므로 오차는 0
    assert calib_err_m == 0.0


def test_compute_widths_reports_disagreeing_scale_error():
    detections = [
        make_detection("승용차", 0.9, x=10, y=10, w=80, h=160),
        make_detection("승용차", 0.9, x=150, y=10, w=120, h=160),
    ]
    _wall_m, _obstacle_m, _effective_m, calib_err_m = compute_widths(
        detections, target_y_px=100.0, camera_height_px=200.0, wall_width_m=4.2
    )
    assert calib_err_m > 0.0


def test_compute_widths_raises_without_reference_vehicle():
    try:
        compute_widths([], target_y_px=0.0, camera_height_px=200.0, wall_width_m=4.2)
    except ValueError:
        pass
    else:
        raise AssertionError("기준 차량이 없으면 에러여야 합니다")


def test_find_narrowest_widths_picks_the_smallest_effective_width():
    # 화면 하단(가까운 곳)에 큰 차, 화면 중간(먼 곳)에 작은 차 — 서로 다른 깊이라
    # 후보 지점마다 obstacle_width_m이 달라진다. 두 후보 중 잔여폭이 더 작은
    # (장애물이 더 큰) 쪽이 "병목"으로 선택돼야 한다.
    near = make_detection("승용차", 0.9, x=10, y=200, w=90, h=180, shape=(400, 300))  # footpoint≈380
    far = make_detection("승용차", 0.9, x=150, y=40, w=30, h=60, shape=(400, 300))  # footpoint≈100

    wall_m, obstacle_m, effective_m, calib_err_m, target_y = find_narrowest_widths(
        [near, far], camera_height_px=400.0, wall_width_m=4.2
    )

    # 두 후보 지점 각각을 직접 계산해서, 실제로 잔여폭이 더 작은 쪽이 선택됐는지 검증
    result_near = compute_widths([near, far], target_y_px=380.0, camera_height_px=400.0, wall_width_m=4.2)
    result_far = compute_widths([near, far], target_y_px=100.0, camera_height_px=400.0, wall_width_m=4.2)
    expected = min([result_near, result_far], key=lambda r: r[2])

    assert wall_m == expected[0]
    assert math.isclose(obstacle_m, expected[1], rel_tol=1e-3)
    assert math.isclose(effective_m, expected[2], rel_tol=1e-3)
    assert math.isclose(calib_err_m, expected[3], rel_tol=1e-3)


def test_find_narrowest_widths_raises_without_detections():
    try:
        find_narrowest_widths([], camera_height_px=200.0, wall_width_m=4.2)
    except ValueError:
        pass
    else:
        raise AssertionError("검출이 없으면 병목 지점을 찾을 수 없어 에러여야 합니다")


def test_vote_status_pass_when_enough_pass_frames():
    assert vote_status(["PASS", "PASS", "PASS", "UNCERTAIN", "FAIL"]) == "PASS"


def test_vote_status_fail_when_majority_fail():
    assert vote_status(["FAIL", "FAIL", "FAIL", "PASS", "UNCERTAIN"]) == "FAIL"


def test_vote_status_uncertain_when_mixed():
    assert vote_status(["PASS", "PASS", "FAIL", "FAIL", "UNCERTAIN"]) == "UNCERTAIN"


def test_vote_status_empty_is_uncertain():
    assert vote_status([]) == "UNCERTAIN"


def test_build_reading_core_shape():
    detections = [
        make_detection("승용차", 0.9, x=10, y=10, w=90, h=180),
    ]
    wall_m, obstacle_m, effective_m, calib_err_m = compute_widths(
        detections, target_y_px=100.0, camera_height_px=200.0, wall_width_m=4.2
    )
    reading = build_reading_core(
        wall_m, obstacle_m, effective_m, detections, VEHICLES_JSON, 0.25, calib_err_m
    )
    assert set(reading.verdict.keys()) == set(VEHICLES_JSON.keys())
    assert all(status in {"PASS", "UNCERTAIN", "FAIL"} for status in reading.verdict.values())
    assert len(reading.detected_objects) == 1
    assert reading.method == "yolov11_homography_v1"
