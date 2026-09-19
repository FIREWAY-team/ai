from __future__ import annotations

import numpy as np

from src import pipeline
from src.pipeline import FrameJudgeJob
from tests.helpers import make_detection

VEHICLES_JSON = {"pump-3.5": 2.3, "pump-8": 2.5}


def test_frame_judge_job_is_edge_id_scoped():
    # v4: verdict가 이미 차종별로 다 담기므로 (edge_id, vehicle_type) 쌍이 아니라
    # edge_id 단위로만 job이 존재한다 (스펙 5장).
    job = FrameJudgeJob(
        edge_id="edge-42",
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=10.0,
        camera_height_px=20.0,
        vehicles_json=VEHICLES_JSON,
    )

    assert job.edge_id == "edge-42"
    assert job.vehicles_json == VEHICLES_JSON


def test_process_frame_uses_map_measured_wall_width(monkeypatch):
    # 지도 실측값(wall_width_m)을 그대로 Reading에 반영한다 — SAM2 자동
    # 추정 없이, 카메라 등록 시 확정한 고정값을 매 프레임 그대로 쓴다.
    detection = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(200, 300))
    monkeypatch.setattr(pipeline.yolo, "detect_vehicles", lambda frame: [detection])

    reading = pipeline.process_frame(
        frame=np.zeros((200, 300, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=100.0,
        camera_height_px=200.0,
        vehicles_json=VEHICLES_JSON,
    )
    assert reading.wall_width_m == 4.2


def test_process_frame_target_y_px_none_uses_bottleneck_search(monkeypatch):
    # target_y_px를 안 주면(어댑터들의 기본값, 2026-09-15부터) 화면 세로 한
    # 지점만 보는 대신 find_narrowest_widths로 병목을 자동 탐색해야 한다 —
    # 실측 검증 중 고정 지점(70%)이 다른 깊이의 진짜 장애물을 놓치는 걸
    # 발견해서 바꿨다.
    near = make_detection("승용차", 0.9, x=10, y=200, w=90, h=180, shape=(400, 300))  # footpoint≈380
    far = make_detection("승용차", 0.9, x=150, y=40, w=30, h=60, shape=(400, 300))  # footpoint≈100
    monkeypatch.setattr(pipeline.yolo, "detect_vehicles", lambda frame: [near, far])

    called_with = {}
    real_find_narrowest = pipeline.find_narrowest_widths

    def _spy_find_narrowest(detections, camera_height_px, wall_width_m, cctv_id=None):
        called_with["used"] = True
        return real_find_narrowest(detections, camera_height_px, wall_width_m, cctv_id=cctv_id)

    monkeypatch.setattr(pipeline, "find_narrowest_widths", _spy_find_narrowest)

    reading = pipeline.process_frame(
        frame=np.zeros((400, 300, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=None,
        camera_height_px=400.0,
        vehicles_json=VEHICLES_JSON,
    )
    assert called_with.get("used") is True
    assert reading.wall_width_m == 4.2


def test_process_frame_explicit_target_y_px_skips_bottleneck_search(monkeypatch):
    # target_y_px를 명시하면(기존 동작) find_narrowest_widths를 안 거치고
    # 그 지점만 그대로 계산한다.
    detection = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(200, 300))
    monkeypatch.setattr(pipeline.yolo, "detect_vehicles", lambda frame: [detection])

    def _boom(*args, **kwargs):
        raise AssertionError("target_y_px를 명시했는데 find_narrowest_widths가 호출됨")

    monkeypatch.setattr(pipeline, "find_narrowest_widths", _boom)

    reading = pipeline.process_frame(
        frame=np.zeros((200, 300, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=190.0,
        camera_height_px=200.0,
        vehicles_json=VEHICLES_JSON,
    )
    assert reading.wall_width_m == 4.2


def test_process_frame_threads_cctv_id_into_compute_widths(monkeypatch):
    # 정확도개선방안 A-4 확장(2026-09-15): cctv_id가 pipeline까지 제대로
    # 전달돼야 누적 캘리브레이션(calibration_store)을 쓸 수 있다.
    detection = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(200, 300))
    monkeypatch.setattr(pipeline.yolo, "detect_vehicles", lambda frame: [detection])

    received = {}

    def fake_compute_widths(detections, target_y_px, camera_height_px, wall_width_m, cctv_id=None):
        received["cctv_id"] = cctv_id
        return wall_width_m, 0.0, wall_width_m, 0.0

    monkeypatch.setattr(pipeline, "compute_widths", fake_compute_widths)

    pipeline.process_frame(
        frame=np.zeros((200, 300, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=100.0,
        camera_height_px=200.0,
        vehicles_json=VEHICLES_JSON,
        cctv_id="cctv_7",
    )
    assert received["cctv_id"] == "cctv_7"


def test_run_job_dispatches_to_process_frame(monkeypatch):
    # process_camera_batch는 ProcessPoolExecutor로 별도 프로세스를 스폰하므로
    # (macOS 기본 spawn) monkeypatch가 자식 프로세스에 전파되지 않는다 — 여기서는
    # 배치 처리가 위임하는 단위인 _run_job만 직접 검증한다.
    detection = make_detection("승용차", 0.9, x=10, y=10, w=90, h=180, shape=(200, 300))
    monkeypatch.setattr(pipeline.yolo, "detect_vehicles", lambda frame: [detection])

    job = FrameJudgeJob(
        edge_id="edge-1",
        frame=np.zeros((200, 300, 3), dtype=np.uint8),
        wall_width_m=4.2,
        target_y_px=100.0,
        camera_height_px=200.0,
        vehicles_json=VEHICLES_JSON,
    )
    reading = pipeline._run_job(job)
    assert reading.wall_width_m == 4.2
