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
