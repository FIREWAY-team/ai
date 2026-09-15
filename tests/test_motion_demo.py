from __future__ import annotations

import numpy as np

import scripts.motion_demo as motion_demo
from scripts.motion_demo import _iou, classify_motion, run_motion_aware_demo, track_vehicles
from tests.helpers import make_detection


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
