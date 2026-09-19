import json
from dataclasses import asdict

import numpy as np
import pytest

from scripts import accumulate_calibration, motion_demo
from src import pipeline
from src.inference import calibration_store, yolo
from src.inference.homography import scale_estimates_with_history
from tests.helpers import make_detection


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("target", [None, 100.])
@pytest.mark.parametrize("confidence", [.5, 1.])
def test_lower_detection_confidence_does_not_turn_blocked_road_into_pass(monkeypatch, video, target, confidence):
    det = make_detection("승용차", confidence, 10, 10, 91, 180)
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: [det])
    frame = np.zeros((200, 300, 3), np.uint8)
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 3.8, target, 200., vehicles_json={"pump-8": 2.5})
    else:
        result = pipeline.process_frame(frame, 3.8, target, 200., vehicles_json={"pump-8": 2.5})
    assert result.obstacle_width_m == pytest.approx(1.8)
    assert result.effective_width_m == pytest.approx(2.)
    assert result.verdict == {"pump-8": "FAIL"}
    assert result.detected_objects[0]["confidence"] == confidence


@pytest.mark.parametrize("legacy_version", ["", "vehicle_width_over_pixels_v2"])
def test_legacy_scale_history_is_never_mixed_with_new_scale(tmp_path, monkeypatch, legacy_version):
    camera = "scale_history_test"
    observation = calibration_store.make_observation(.01, .5, 189, 200)
    legacy = tmp_path / legacy_version / f"{camera}.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps(asdict(observation)) + "\n")
    original = legacy.read_bytes()
    assert calibration_store.load_observations(camera, calibration_dir=tmp_path) == []

    det = make_detection("승용차", .5, 10, 10, 91, 180)
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: [det])
    monkeypatch.setattr(accumulate_calibration, "load_frame", lambda _: np.zeros((200, 300, 3), np.uint8))
    assert accumulate_calibration.accumulate_camera(camera, "demo.png", calibration_dir=tmp_path) == 1
    loader = calibration_store.load_observations
    monkeypatch.setattr(calibration_store, "load_observations", lambda camera: loader(camera, calibration_dir=tmp_path))
    # 현재 프레임에 기준 차량이 없어도 보정된 누적 스케일만 사용한다.
    assert scale_estimates_with_history([], 200, camera) == [(.02, .5, 189.)]
    assert legacy.read_bytes() == original
    stored = tmp_path / calibration_store.SCALE_VERSION / f"{camera}.jsonl"
    assert json.loads(stored.read_text())["scale"] == pytest.approx(.02)
