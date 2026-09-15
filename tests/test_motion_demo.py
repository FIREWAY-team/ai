from __future__ import annotations

import numpy as np
import pytest

from src.inference import calibration_store

import scripts.motion_demo as motion_demo
from scripts.motion_demo import _iou, classify_motion, run_motion_aware_demo, track_vehicles
from tests.helpers import make_detection


@pytest.mark.parametrize("target_y", [None, 190.])
@pytest.mark.parametrize("wall_width, expected", [(3.8, "FAIL"), (8., "PASS")])
def test_new_vehicle_counts_as_obstacle_without_assuming_motion(monkeypatch, target_y, wall_width, expected):
    car = make_detection("승용차", .9, 10, 10, 90, 180)
    detections = iter([[], [car]])
    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", lambda frame: next(detections))
    result = run_motion_aware_demo(
        [np.zeros((200, 300, 3), dtype=np.uint8)] * 2,
        wall_width, target_y, 200., vehicles_json={"pump-8": 2.5},
    )
    assert result.obstacle_width_m == pytest.approx(1.8)
    assert result.effective_width_m == pytest.approx(wall_width - 1.8)
    assert result.verdict == {"pump-8": expected}


@pytest.mark.parametrize("target_y", [None, 190.])
def test_moving_vehicle_does_not_hide_new_unknown_vehicle(monkeypatch, target_y):
    shape = (300, 400)
    moving_before = make_detection("승용차", .9, 10, 10, 90, 180, shape=shape)
    moving_after = make_detection("승용차", .9, 10, 40, 90, 180, shape=shape)
    new_car = make_detection("승용차", .9, 220, 10, 90, 180, shape=shape)
    detections = iter([[moving_before], [moving_after, new_car]])
    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", lambda frame: next(detections))
    result = run_motion_aware_demo(
        [np.zeros((*shape, 3), dtype=np.uint8)] * 2,
        3.8, target_y, 300., vehicles_json={"pump-8": 2.5},
    )
    assert result.obstacle_width_m == pytest.approx(1.8)
    assert result.verdict == {"pump-8": "FAIL"}


@pytest.mark.parametrize("target_y", [None, 190.])
def test_motion_uses_history_when_last_frame_has_no_reference(monkeypatch, target_y):
    person = make_detection("보행자", .9, 10, 10, 90, 180)
    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", lambda frame: [person])
    monkeypatch.setattr(calibration_store, "load_observations", lambda camera: [
        calibration_store.make_observation(.02, .9, 190., 200.),
        calibration_store.make_observation(.03, .9, 190., 200.),
    ])
    result = run_motion_aware_demo(
        [np.zeros((200, 300, 3), dtype=np.uint8)] * 2, 4.2, target_y, 200., cctv_id="alias",
    )
    assert result.calibration_error_m == pytest.approx(.84)
    assert result.obstacle_width_m == 0.


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert _iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert _iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_classify_motion_stationary_when_no_displacement():
    tracks = {0: [(10.0, 10.0, 0), (10.0, 10.0, 1)]}
    result = classify_motion(tracks, scale_m_per_px=0.02)
    assert result[0] == "STATIONARY"


def test_classify_motion_moving_when_fast_displacement():
    # 1초에 100px 이동, 스케일 0.02m/px → 2m/s (임계값 0.3m/s 초과)
    tracks = {0: [(0.0, 0.0, 0), (100.0, 0.0, 1)]}
    result = classify_motion(tracks, scale_m_per_px=0.02, frame_interval_sec=1.0)
    assert result[0] == "MOVING"


def test_classify_motion_unknown_with_single_observation():
    tracks = {0: [(0.0, 0.0, 0)]}
    result = classify_motion(tracks, scale_m_per_px=0.02)
    assert result[0] == "UNKNOWN"


@pytest.mark.parametrize("interval, expected", [(1., "STATIONARY"), (.5, "STATIONARY"), (.1, "MOVING")])
def test_motion_uses_frame_indices_across_missing_detections(interval, expected):
    tracks = {0: [(0., 0., 2), (50., 0., 11)]}
    assert classify_motion(tracks, .02, frame_interval_sec=interval)[0] == expected


@pytest.mark.parametrize("indices", [(2, 2), (5, 2), (2, 4, 3, 5)])
def test_motion_unknown_when_observation_times_do_not_increase(indices):
    tracks = {0: [(float(i * 50), 0., frame_idx) for i, frame_idx in enumerate(indices)]}
    assert classify_motion(tracks, .02)[0] == "UNKNOWN"


@pytest.mark.parametrize("interval", [0., -1., float("nan"), float("inf")])
def test_motion_rejects_invalid_frame_interval(interval):
    with pytest.raises(ValueError, match="프레임 간격"):
        classify_motion({0: [(0., 0., 0), (50., 0., 1)]}, .02, frame_interval_sec=interval)


@pytest.mark.parametrize("target_y", [None, 190.])
def test_vehicle_reappearing_after_detection_gap_remains_an_obstacle(monkeypatch, target_y):
    first = make_detection("승용차", .9, 10, 10, 90, 180)
    last = make_detection("승용차", .9, 40, 10, 90, 180)
    detections = iter([[first]] + [[]] * 8 + [[last]])
    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", lambda frame: next(detections))
    result = run_motion_aware_demo(
        [np.zeros((200, 300, 3), dtype=np.uint8)] * 10,
        3.8, target_y, 200., vehicles_json={"pump-8": 2.5},
    )
    assert result.obstacle_width_m == pytest.approx(1.8)
    assert result.verdict == {"pump-8": "FAIL"}


def test_track_vehicles_matches_same_vehicle_across_frames():
    frame1 = [make_detection("승용차", 0.9, x=50, y=80, w=90, h=180)]
    # 다음 프레임에서 살짝 이동(정지 차량 노이즈 수준)
    frame2 = [make_detection("승용차", 0.9, x=52, y=80, w=90, h=180)]
    tracks, last_assignment = track_vehicles([frame1, frame2])
    assert len(tracks) == 1  # 같은 차량으로 매칭돼야 한다
    assert len(last_assignment) == 1


def test_track_vehicles_separates_distinct_vehicles():
    frame1 = [
        make_detection("승용차", 0.9, x=10, y=10, w=40, h=80, shape=(200, 400)),
        make_detection("승용차", 0.9, x=300, y=10, w=40, h=80, shape=(200, 400)),
    ]
    tracks, last_assignment = track_vehicles([frame1])
    assert len(tracks) == 2
    assert len(last_assignment) == 2


def test_motion_demo_requires_at_least_two_frames():
    try:
        run_motion_aware_demo(
            [np.zeros((20, 20, 3), dtype=np.uint8)],
            wall_width_m=4.2,
            target_y_px=10.0,
            camera_height_px=20.0,
            vehicles_json={"pump-3.5": 2.3, "pump-8": 2.5},
        )
    except ValueError as error:
        assert str(error) == "최소 2개 이상의 프레임이 필요합니다"
    else:
        raise AssertionError("단일 프레임 데모는 ValueError를 발생해야 합니다")


def test_motion_demo_target_y_px_none_uses_bottleneck_search(monkeypatch):
    # 2026-09-15: target_y_px를 안 주면 find_narrowest_widths로 정지 차량들
    # 중 병목 지점을 자동으로 찾아야 한다 — 영상 경로도 이미지 경로
    # (pipeline.process_frame)와 같은 방식으로 바뀜.
    shape = (400, 300)
    ref = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=shape)  # 기준 차량, footpoint=190
    obstacle_a = make_detection("승용차", 0.9, x=150, y=200, w=60, h=120, shape=shape)  # footpoint=320, 정지
    obstacle_b_frame1 = make_detection("승용차", 0.9, x=150, y=202, w=60, h=120, shape=shape)  # 거의 제자리(노이즈)

    per_frame = [[ref, obstacle_a], [ref, obstacle_b_frame1]]
    call_count = {"n": 0}

    def fake_detect(frame):
        idx = min(call_count["n"], len(per_frame) - 1)
        call_count["n"] += 1
        return per_frame[idx]

    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", fake_detect)

    frames = [np.zeros((*shape, 3), dtype=np.uint8) for _ in range(2)]
    reading = run_motion_aware_demo(
        frames,
        wall_width_m=4.2,
        target_y_px=None,
        camera_height_px=float(shape[0]),
        vehicles_json={"pump-3.5": 2.3, "pump-8": 2.5},
    )
    # 정지 장애물(obstacle_a)이 병목탐색으로 잡혀서 obstacle_width_m > 0이어야 한다
    assert reading.obstacle_width_m > 0.0
    assert reading.effective_width_m < 4.2


def test_motion_demo_threads_cctv_id_into_find_narrowest_widths(monkeypatch):
    # 2026-09-15: 이미지 경로(pipeline.process_frame)는 이미 cctv_id를 받아
    # 누적 캘리브레이션(calibration_store)을 쓰는데, 영상 경로는 빠져 있었다
    # — scripts/accumulate_calibration.py로 영상 카메라에도 관측치를 쌓아놨지만
    # 실제로는 안 쓰이던 버그.
    shape = (400, 300)
    ref = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=shape)
    obstacle_a = make_detection("승용차", 0.9, x=150, y=200, w=60, h=120, shape=shape)
    per_frame = [[ref, obstacle_a], [ref, obstacle_a]]
    call_count = {"n": 0}

    def fake_detect(frame):
        idx = min(call_count["n"], len(per_frame) - 1)
        call_count["n"] += 1
        return per_frame[idx]

    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", fake_detect)

    received = {}
    real_find_narrowest = motion_demo.find_narrowest_widths

    def _spy_find_narrowest(detections, camera_height_px, wall_width_m, cctv_id=None):
        received["cctv_id"] = cctv_id
        return real_find_narrowest(detections, camera_height_px, wall_width_m, cctv_id=cctv_id)

    monkeypatch.setattr(motion_demo, "find_narrowest_widths", _spy_find_narrowest)

    frames = [np.zeros((*shape, 3), dtype=np.uint8) for _ in range(2)]
    run_motion_aware_demo(
        frames,
        wall_width_m=4.2,
        target_y_px=None,
        camera_height_px=float(shape[0]),
        vehicles_json={"pump-3.5": 2.3, "pump-8": 2.5},
        cctv_id="cctv_4",
    )
    assert received["cctv_id"] == "cctv_4"


def test_motion_demo_target_y_px_none_with_no_stationary_vehicles_is_clear(monkeypatch):
    # 정지 차량이 하나도 없으면(전부 이동 중) 병목탐색을 시도할 대상 자체가
    # 없다 — 도로가 뚫려있는 것으로 취급(obstacle=0).
    #
    # 주의: 스케일 계산(local_scale_estimates)은 last_detections 전체(정지
    # 여부 무관)를 보므로, 기준 차량이 "이동 중"으로 분류돼도 캘리브레이션
    # 자체는 여전히 성립한다 — 이 테스트는 그 기준 차량 한 대가 프레임 간
    # 충분히 크게(하지만 IoU 매칭은 유지될 만큼) 이동해서 STATIONARY가 아닌
    # MOVING으로 분류되는 상황을 만든다. (이전 버전은 기준 차량을 두 프레임
    # 모두 같은 자리에 둬서 실제로는 정지 차량이 존재하는 셈이었다 — 테스트
    # 설계 버그.)
    shape = (400, 300)
    ref_frame1 = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=shape)
    ref_frame2 = make_detection("승용차", 0.9, x=10, y=40, w=90, h=180, shape=shape)  # 30px 이동(IoU 유지, 속도 임계 초과)

    per_frame = [[ref_frame1], [ref_frame2]]
    call_count = {"n": 0}

    def fake_detect(frame):
        idx = min(call_count["n"], len(per_frame) - 1)
        call_count["n"] += 1
        return per_frame[idx]

    monkeypatch.setattr(motion_demo.yolo, "detect_vehicles", fake_detect)

    frames = [np.zeros((*shape, 3), dtype=np.uint8) for _ in range(2)]
    reading = run_motion_aware_demo(
        frames,
        wall_width_m=4.2,
        target_y_px=None,
        camera_height_px=float(shape[0]),
        vehicles_json={"pump-3.5": 2.3, "pump-8": 2.5},
    )

    assert reading.obstacle_width_m == 0.0
    assert reading.effective_width_m == 4.2
