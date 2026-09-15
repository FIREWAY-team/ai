"""입력 어댑터 — HTTP still_url (스펙 0장). CCTV 관제 서버가 내려주는
정지 이미지 URL(예: 로드뷰 캡처를 서빙하는 내부 스토리지)에서 한 장을
받아온다. 내부 판독 로직은 file_adapter와 동일 — 프레임 획득 방식만 다르다.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import requests

from src import camera_registry
from src.adapters.common import build_reading
from src.pipeline import process_frame
from src.schemas import ReadingCore

REQUEST_TIMEOUT_SEC = 10


def fetch_frame(still_url: str, timeout_sec: int = REQUEST_TIMEOUT_SEC):
    """HTTP(S) still_url → BGR np.ndarray 프레임."""
    response = requests.get(still_url, timeout=timeout_sec)
    response.raise_for_status()
    buffer = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"이미지를 디코딩할 수 없습니다: {still_url}")
    return frame


def read_from_http(
    still_url: str,
    cctv_id: str,
    wall_width_m: float | None = None,
    target_y_px: float | None = None,
    camera_height_px: float | None = None,
    edge_id: str | None = None,
    vehicles_json: dict[str, float] | None = None,
    margin_m: float | None = None,
) -> dict[str, Any]:
    """HTTP still_url 한 장 → Reading(dict). wall_width_m을 직접 안 주면
    `configs/cameras.yaml`에서 cctv_id로 조회한다(src.camera_registry).
    target_y_px를 안 주면(기본) process_frame이 병목 지점을 자동으로 찾는다
    (find_narrowest_widths — file_adapter.read_from_file 참고). camera_height_px도
    안 주면 이미지 크기에서 잡는다."""
    if wall_width_m is None:
        wall_width_m = camera_registry.get_camera(cctv_id)["wall_width_m"]
    frame = fetch_frame(still_url)
    if camera_height_px is None:
        camera_height_px = frame.shape[0]
    reading_core: ReadingCore = process_frame(
        frame,
        wall_width_m,
        target_y_px,
        camera_height_px,
        vehicles_json=vehicles_json,
        margin_m=margin_m,
        cctv_id=cctv_id,
    )
    return build_reading(
        cctv_id=cctv_id,
        edge_id=edge_id,
        still_url=still_url,
        reading_core=reading_core,
        source_meta={"adapter": "http"},
    )
