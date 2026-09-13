"""입력 어댑터 — RTSP 스트림 (스펙 0장). 실시간 CCTV 스트림에서 프레임
한 장을 캡처해 나머지는 file/http 어댑터와 동일한 정지 이미지 경로를 탄다
("RTSP로 바뀌어도 내부 로직은 동일" — 스펙 1장).
"""
from __future__ import annotations

from typing import Any

import cv2

from src import camera_registry
from src.adapters.common import build_reading, default_target_y_px
from src.pipeline import process_frame
from src.schemas import ReadingCore

CONNECT_TIMEOUT_MS = 5000


def capture_frame(rtsp_url: str, connect_timeout_ms: int = CONNECT_TIMEOUT_MS):
    """RTSP 스트림 → 최신 프레임 한 장(BGR np.ndarray)."""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, connect_timeout_ms)
    try:
        if not cap.isOpened():
            raise ValueError(f"RTSP 스트림에 연결할 수 없습니다: {rtsp_url}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"RTSP 스트림에서 프레임을 읽지 못했습니다: {rtsp_url}")
        return frame
    finally:
        cap.release()


def read_from_rtsp(
    rtsp_url: str,
    cctv_id: str,
    wall_width_m: float | None = None,
    target_y_px: float | None = None,
    camera_height_px: float | None = None,
    edge_id: str | None = None,
    vehicles_json: dict[str, float] | None = None,
    margin_m: float | None = None,
) -> dict[str, Any]:
    """RTSP 스트림에서 프레임 한 장 캡처 → Reading(dict). wall_width_m을 직접
    안 주면 `configs/cameras.yaml`에서 cctv_id로 조회한다(src.camera_registry).
    target_y_px/camera_height_px도 안 주면 이미지 크기에서 근사치를 잡는다."""
    if wall_width_m is None:
        wall_width_m = camera_registry.get_camera(cctv_id)["wall_width_m"]
    frame = capture_frame(rtsp_url)
    if camera_height_px is None:
        camera_height_px = frame.shape[0]
    if target_y_px is None:
        target_y_px = default_target_y_px(frame)
    reading_core: ReadingCore = process_frame(
        frame,
        wall_width_m,
        target_y_px,
        camera_height_px,
        vehicles_json=vehicles_json,
        margin_m=margin_m,
    )
    return build_reading(
        cctv_id=cctv_id,
        edge_id=edge_id,
        still_url=None,
        reading_core=reading_core,
        source_meta={"adapter": "rtsp", "rtsp_url": rtsp_url},
    )
