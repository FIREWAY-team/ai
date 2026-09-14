"""입력 어댑터 — 로컬 이미지 파일 (스펙 0장). 정지 이미지 프레임 한 장을
읽어 파이프라인에 넘기는 가장 단순한 경로. RTSP/HTTP 어댑터도 프레임을
얻는 방식만 다를 뿐 내부 로직은 동일하다(스펙 1장).
"""
from __future__ import annotations

from typing import Any

import cv2

from src import camera_registry
from src.adapters.common import build_reading, default_target_y_px
from src.pipeline import process_frame
from src.schemas import ReadingCore


def load_frame(path: str):
    """이미지 파일 경로 → BGR np.ndarray 프레임."""
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return frame


def read_from_file(
    path: str,
    cctv_id: str,
    wall_width_m: float | None = None,
    target_y_px: float | None = None,
    camera_height_px: float | None = None,
    edge_id: str | None = None,
    vehicles_json: dict[str, float] | None = None,
    margin_m: float | None = None,
) -> dict[str, Any]:
    """파일 한 장 → Reading(dict). 판독 로직은 src.pipeline.process_frame에 위임한다.

    wall_width_m을 직접 안 주면 `configs/cameras.yaml`에서 cctv_id로 조회한다
    (카메라 등록 시 지도 실측으로 확정한 값 — src.camera_registry). target_y_px/
    camera_height_px도 안 주면 이미지 크기에서 근사치를 잡는다(정확도가
    중요하면 명시적으로 넘길 것 — src.adapters.common.default_target_y_px).
    """
    if wall_width_m is None:
        wall_width_m = camera_registry.get_camera(cctv_id)["wall_width_m"]
    frame = load_frame(path)
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
        cctv_id=cctv_id,
    )
    return build_reading(
        cctv_id=cctv_id,
        edge_id=edge_id,
        still_url=path,
        reading_core=reading_core,
        source_meta={"adapter": "file"},
    )
