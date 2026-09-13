from __future__ import annotations

import numpy as np

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
