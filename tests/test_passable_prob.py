from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from goldenlane_vehicle_specs import (
    MIN_MARGIN_M,
    resolve_margin_m,
    resolve_margin_m_from_samples,
)
from src.inference import calibration_store
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


def test_vehicles_json_matches_backend_fleet():
    from goldenlane_vehicle_specs import load_vehicles_json

    vehicles = load_vehicles_json()
    assert vehicles == {
        "pump-3.5": 2.3, "pump-8": 2.5, "pump-15": 2.9, "aerial-25": 2.5,
    }
    results, _ = build_reading_verdict(3.0, vehicles, 0.25)
    assert results == {
        "pump-3.5": "PASS", "pump-8": "PASS",
        "pump-15": "UNCERTAIN", "aerial-25": "PASS",
    }


def irregular_vehicle():
    det = make_detection("승용차", .9, 60, 70, 151, 191, shape=(300, 300))
    mask = np.zeros((300, 300), dtype=np.uint8)
    contour = np.array([[160, 70], [210, 70], [210, 140], [100, 260], [60, 260], [60, 200]])
    cv2.fillPoly(mask, [contour], 1)
    det.mask = mask.astype(bool)
    return det


def test_empty_corner_of_rotated_box_does_not_block_road():
    det = irregular_vehicle()
    assert not det.mask[58:63].any()
    assert obstacle_widths_m([det], .02, target_y_px=60., camera_height_px=300.) == 0.


def test_hole_between_mask_rows_does_not_block_road():
    det = make_detection("승용차", .9, 10, 10, 90, 180)
    det.mask[90:111] = False
    assert obstacle_widths_m([det], .02, target_y_px=100., camera_height_px=200.) == 0.


def test_bottleneck_candidate_still_intersects_irregular_vehicle_mask():
    det = irregular_vehicle()
    _, obstacle, _, _, target = find_narrowest_widths([det], 300., 4.2)
    assert obstacle == pytest.approx(1.8)
    assert det.mask[int(round(target))].any()


def test_fractional_target_uses_existing_pixel_tolerance():
    det = make_detection("승용차", .9, 10, 10, 90, 180)
    assert obstacle_widths_m([det], .02, 190., 200.) == pytest.approx(89 * .02)


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


def test_verdict_high_calibration_error_downgrades_marginal_pass_to_uncertain():
    # 좁은 골목을 비스듬히 찍은 카메라는 기준 차량들의 스케일 추정치가 서로
    # 크게 어긋날 수 있다(2026-09-13 실측 검증에서 확인 — 같은 프레임 안에서
    # 기준 차량 스케일이 최대 2배 차이). 그런 프레임에서는 effective_width_m이
    # margin_m 기준으로 PASS처럼 보여도, 그 자체가 불확실한 값이므로 confident
    # PASS를 내면 안 된다.
    need = VEHICLES_JSON["pump-3.5"]
    margin_m = 0.25
    effective_m = need + margin_m  # calibration_error_m=0이면 z==1, PASS
    status_confident, _ = verdict(effective_m, need, margin_m, calibration_error_m=0.0)
    assert status_confident == "PASS"

    status_uncertain, _ = verdict(effective_m, need, margin_m, calibration_error_m=1.0)
    assert status_uncertain == "UNCERTAIN"


def test_verdict_high_calibration_error_does_not_soften_fail():
    # 오류 비대칭성(스펙 4장): FAIL을 PASS로 오판하는 쪽이 훨씬 위험하므로,
    # 캘리브레이션이 불확실하다고 해서 이미 확정된 FAIL 판정을 UNCERTAIN으로
    # 완화하지 않는다 — PASS 쪽만 더 보수적으로 넓어진다.
    need = VEHICLES_JSON["pump-3.5"]
    margin_m = 0.25
    effective_m = need - margin_m  # calibration_error_m=0이면 z==-1, FAIL
    status, _ = verdict(effective_m, need, margin_m, calibration_error_m=5.0)
    assert status == "FAIL"


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


def test_obstacle_widths_m_same_lane_overlap_takes_max_not_sum():
    # 2026-09-15, cctv_4 실측 검증 중 발견: 같은 차선에 앞뒤로 거의 붙어
    # 주차된 두 차량은 원근 압축 때문에 vehicle_y_span이 경계에서 몇 px
    # 겹칠 수 있다 — 이때 가로 위치(vehicle_x_span)까지 겹치면(같은 차선)
    # 나란히 서서 폭을 나눠 막는 게 아니므로 더하지 않고 더 넓은 차 1대분만
    # 반영해야 한다. (가로 위치가 안 겹치는 다른 차선 케이스는
    # test_obstacle_widths_m_sums_pixel_widths_scaled가 이미 검증한다.)
    near_car = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(400, 300))  # y:10~189, x:10~99
    far_car = make_detection("승용차", 0.9, x=30, y=185, w=90, h=60, shape=(400, 300))  # y:185~244, x:30~119(겹침)

    total = obstacle_widths_m(
        [near_car, far_car], scale_m_per_px=0.02, target_y_px=187.0, camera_height_px=400.0
    )
    near_only = obstacle_widths_m(
        [near_car], scale_m_per_px=0.02, target_y_px=187.0, camera_height_px=400.0
    )
    far_only = obstacle_widths_m(
        [far_car], scale_m_per_px=0.02, target_y_px=187.0, camera_height_px=400.0
    )
    # 더 넓은 차(near_car) 1대분과 같아야 하고, 두 폭을 더한 값보다는 작아야 한다.
    assert math.isclose(total, near_only, rel_tol=1e-6)
    assert total < near_only + far_only


def test_obstacle_widths_m_excludes_people():
    # 보행자는 소방차가 오면 스스로 비켜설 수 있으므로 도로 폭 계산에서
    # 제외한다(정책 확정) — 검출은 되지만 폭 합산에는 안 들어간다.
    car = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180)  # footpoint=190
    person = make_detection("보행자", 0.8, x=150, y=10, w=40, h=180)  # 같은 깊이(footpoint=190)

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


def test_compute_widths_without_cctv_id_never_touches_calibration_store(monkeypatch):
    # cctv_id를 안 주면(기존 호출부, 하위호환) calibration_store는 아예 조회되지
    # 않아야 한다 — 조회하면 즉시 실패하게 만들어서 검증한다.
    def _boom(*args, **kwargs):
        raise AssertionError("cctv_id 없이 호출했는데 calibration_store가 조회됨")

    monkeypatch.setattr(calibration_store, "load_observations", _boom)

    detections = [make_detection("승용차", 0.9, x=10, y=10, w=90, h=180)]
    compute_widths(detections, target_y_px=190.0, camera_height_px=200.0, wall_width_m=4.2)


def test_compute_widths_uses_accumulated_observations_when_current_frame_lacks_references(
    monkeypatch,
):
    # 정확도개선방안 A-4 확장(2026-09-15): 현재 프레임에 기준 차량이 하나도
    # 없어도, cctv_id로 누적된 과거 관측치가 있으면 그걸로 스케일을 계산할 수
    # 있어야 한다. 장애물(사람)만 있고 기준 차량은 없는 프레임으로 검증.
    person = make_detection("보행자", 0.9, x=10, y=10, w=90, h=180)  # 기준자 후보 아님

    monkeypatch.setattr(
        calibration_store,
        "load_observations",
        lambda cctv_id, **kwargs: [
            calibration_store.make_observation(0.02, 0.9, 190.0, 200.0),
            calibration_store.make_observation(0.021, 0.85, 185.0, 200.0),
        ],
    )

    # cctv_id 없이는(현재 프레임에 기준 차량이 없으므로) 에러가 나야 한다
    try:
        compute_widths([person], target_y_px=190.0, camera_height_px=200.0, wall_width_m=4.2)
    except ValueError:
        pass
    else:
        raise AssertionError("기준 차량이 없으면 cctv_id 없이는 에러여야 합니다")

    # cctv_id를 주면 누적 관측치로 계산에 성공해야 한다
    wall_m, obstacle_m, effective_m, calib_err_m = compute_widths(
        [person], target_y_px=190.0, camera_height_px=200.0, wall_width_m=4.2, cctv_id="cctv_x"
    )
    assert wall_m == 4.2
    assert effective_m <= wall_m


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
    assert reading.method == "yolov11_homography_v7"
