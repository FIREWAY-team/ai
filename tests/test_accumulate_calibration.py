from __future__ import annotations

import numpy as np
import pytest

import scripts.accumulate_calibration as accumulate_calibration
from src.inference import calibration_store
from tests.helpers import make_detection


def test_accumulate_camera_routes_video_through_extract_frames(monkeypatch, tmp_path):
    # still_url이 동영상이면 이미지 1장이 아니라 extract_frames로 뽑은 여러
    # 프레임 각각에서 기준 차량을 찾아야 한다 — 화면을 가로지르는 차량은
    # 프레임마다 다른 깊이(footpoint_y)를 주므로, 한 번의 실행만으로도
    # 이미지 카메라보다 훨씬 촘촘한 깊이별 관측치를 쌓을 수 있다.
    fake_frames = [np.zeros((200, 300, 3), dtype=np.uint8) for _ in range(3)]
    monkeypatch.setattr(
        accumulate_calibration, "extract_frames", lambda path, max_frames=10: fake_frames
    )

    footpoints = iter([190.0, 150.0, 110.0])

    def fake_detect(frame):
        y = next(footpoints)
        # y=190일 때 footpoint가 y=190 근처가 되도록 세로로 긴 사각형 하나만 배치
        h = int(y) - 10
        return [make_detection("승용차", 0.9, x=10, y=10, w=90, h=h, shape=(200, 300))]

    monkeypatch.setattr(accumulate_calibration.yolo, "detect_vehicles", fake_detect)

    saved = accumulate_calibration.accumulate_camera("cctv_video", "clip.mp4", calibration_dir=tmp_path)

    assert saved == 3
    loaded = calibration_store.load_observations("cctv_video", calibration_dir=tmp_path)
    assert len(loaded) == 3
    footpoint_ys = {round(obs.footpoint_y) for obs in loaded}
    assert len(footpoint_ys) == 3  # 프레임마다 서로 다른 깊이가 쌓였어야 한다


def test_accumulate_camera_uses_single_frame_for_local_image(monkeypatch, tmp_path):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    monkeypatch.setattr(accumulate_calibration, "load_frame", lambda path: frame)

    detection = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(200, 300))
    monkeypatch.setattr(accumulate_calibration.yolo, "detect_vehicles", lambda f: [detection])

    saved = accumulate_calibration.accumulate_camera("cctv_img", "still.png", calibration_dir=tmp_path)

    assert saved == 1
    loaded = calibration_store.load_observations("cctv_img", calibration_dir=tmp_path)
    assert len(loaded) == 1


def test_accumulate_camera_returns_zero_when_no_reference_vehicles(monkeypatch, tmp_path):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    monkeypatch.setattr(accumulate_calibration, "load_frame", lambda path: frame)
    monkeypatch.setattr(accumulate_calibration.yolo, "detect_vehicles", lambda f: [])

    saved = accumulate_calibration.accumulate_camera("cctv_empty", "still.png", calibration_dir=tmp_path)

    assert saved == 0
    assert calibration_store.load_observations("cctv_empty", calibration_dir=tmp_path) == []


def test_accumulate_all_processes_shared_footage_once(monkeypatch):
    cameras = {
        "alias": {"still_url": "clip.mp4", "calibration_source_cctv_id": "original"},
        "original": {"still_url": "clip.mp4"},
    }
    monkeypatch.setattr(accumulate_calibration.camera_registry, "load_cameras", lambda *args: cameras)
    calls = []
    monkeypatch.setattr(accumulate_calibration, "accumulate_camera",
                        lambda camera, url: calls.append((camera, url)) or 1)
    accumulate_calibration.accumulate_all()
    assert calls == [("alias", "clip.mp4")]


def test_accumulate_alias_rejects_other_input_before_loading_frames(monkeypatch, tmp_path):
    cameras = {
        "alias": {"still_url": "clip.mp4", "calibration_source_cctv_id": "original"},
        "original": {"still_url": "clip.mp4"},
    }
    monkeypatch.setattr(accumulate_calibration.camera_registry, "load_cameras", lambda *args: cameras)
    monkeypatch.setattr(accumulate_calibration, "_load_frames_for_still_url",
                        lambda url: pytest.fail("다른 원본 프레임 로드"))
    with pytest.raises(ValueError, match="다른 영상"):
        accumulate_calibration.accumulate_camera("alias", "other.mp4", calibration_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
