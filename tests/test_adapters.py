from __future__ import annotations

import numpy as np
import pytest

from src import camera_registry
from src.adapters import file_adapter, http_adapter, rtsp_adapter
from src.schemas import ReadingCore

FAKE_READING_CORE = ReadingCore(
    wall_width_m=4.2,
    obstacle_width_m=1.0,
    effective_width_m=3.2,
    detected_objects=[],
    verdict={"pump-3.5": "PASS", "pump-8": "UNCERTAIN"},
    confidence=0.7,
    calibration_error_m=0.05,
)


def test_read_from_file_builds_reading(monkeypatch, tmp_path):
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(file_adapter, "load_frame", lambda path: fake_frame)
    monkeypatch.setattr(file_adapter, "process_frame", lambda *args, **kwargs: FAKE_READING_CORE)

    image_path = str(tmp_path / "frame.jpg")
    reading = file_adapter.read_from_file(
        image_path,
        cctv_id="cam_l1",
        wall_width_m=4.2,
        target_y_px=5.0,
        camera_height_px=10.0,
        edge_id="edge-1",
    )

    assert reading["cctv_id"] == "cam_l1"
    assert reading["edge_id"] == "edge-1"
    assert reading["still_public_url"] == image_path
    assert reading["wall_width_m"] == 4.2
    assert reading["verdict"] == {"pump-3.5": "PASS", "pump-8": "UNCERTAIN"}
    assert reading["source_meta"] == {"adapter": "file"}


def test_load_frame_raises_for_missing_file(tmp_path):
    with pytest.raises(ValueError):
        file_adapter.load_frame(str(tmp_path / "does_not_exist.jpg"))


def test_read_from_http_builds_reading(monkeypatch):
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(http_adapter, "fetch_frame", lambda url, timeout_sec=10: fake_frame)
    monkeypatch.setattr(http_adapter, "process_frame", lambda *args, **kwargs: FAKE_READING_CORE)

    reading = http_adapter.read_from_http(
        "https://example.internal/cam1/still.jpg",
        cctv_id="cam_l1",
        wall_width_m=4.2,
        target_y_px=5.0,
        camera_height_px=10.0,
    )

    assert reading["still_public_url"] == "https://example.internal/cam1/still.jpg"
    assert reading["source_meta"] == {"adapter": "http"}
    assert reading["effective_width_m"] == 3.2


def test_read_from_file_looks_up_wall_width_m_by_cctv_id(monkeypatch, tmp_path):
    # wall_width_m을 안 주면 configs/cameras.yaml(camera_registry)에서 cctv_id로 조회한다.
    monkeypatch.setattr(
        camera_registry, "get_camera", lambda cctv_id: {"wall_width_m": 4.2}
    )

    fake_frame = np.zeros((20, 30, 3), dtype=np.uint8)
    monkeypatch.setattr(file_adapter, "load_frame", lambda path: fake_frame)
    captured = {}

    def _fake_process_frame(frame, wall_width_m, target_y_px, camera_height_px, **kwargs):
        captured["wall_width_m"] = wall_width_m
        captured["target_y_px"] = target_y_px
        captured["camera_height_px"] = camera_height_px
        return FAKE_READING_CORE

    monkeypatch.setattr(file_adapter, "process_frame", _fake_process_frame)

    file_adapter.read_from_file(str(tmp_path / "frame.jpg"), cctv_id="cam_l1")

    assert captured["wall_width_m"] == 4.2
    assert captured["camera_height_px"] == 20
    # target_y_px를 명시 안 하면 process_frame이 병목 지점을 자동으로 찾도록
    # None을 그대로 넘긴다(2026-09-15부터 — find_narrowest_widths 참고). 화면의
    # 고정 비율 지점만 보면 다른 깊이의 진짜 장애물을 놓칠 수 있어서 바뀌었다.
    assert captured["target_y_px"] is None


def test_read_from_rtsp_builds_reading(monkeypatch):
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(rtsp_adapter, "capture_frame", lambda url, connect_timeout_ms=5000: fake_frame)
    monkeypatch.setattr(rtsp_adapter, "process_frame", lambda *args, **kwargs: FAKE_READING_CORE)

    reading = rtsp_adapter.read_from_rtsp(
        "rtsp://cam.internal/stream1",
        cctv_id="cam_l1",
        wall_width_m=4.2,
        target_y_px=5.0,
        camera_height_px=10.0,
    )

    assert reading["still_public_url"] is None
    assert reading["source_meta"]["adapter"] == "rtsp"
    assert reading["source_meta"]["rtsp_url"] == "rtsp://cam.internal/stream1"
