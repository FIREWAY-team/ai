"""세 어댑터(file/http/rtsp) 공용 — ReadingCore를 팀 공용 Reading(dict)으로 감싼다 (스펙 0, 1장).

Reading 전체 스키마(cctv_id, still_public_url, edge_id, measured_at, source_meta
등)는 백엔드(`FIREWAY-team/backend`)의 `cctv_readings` 테이블(V3 마이그레이션)
소유라 이 저장소에 dataclass로 재정의하지 않는다 — 그 테이블 컬럼명 그대로
dict로 채워 넘긴다. 판독 모듈이 책임지는 하위 집합(ReadingCore,
src/schemas.py)만 이 저장소가 소유한다.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.schemas import ReadingCore

# 장애물 위치(target_y_px)를 카메라마다 수동 지정하지 않을 때 쓰는 기본값 —
# 도로면은 보통 화면 하단부에 걸쳐 있으므로 세로 70% 지점을 기본 타깃으로
# 잡는다. 데모용 근사치이며, 카메라별 정확도가 중요해지면 명시적으로
# target_y_px를 넘겨서 override한다.
DEFAULT_TARGET_Y_RATIO = 0.7


def default_target_y_px(frame: np.ndarray) -> float:
    return frame.shape[0] * DEFAULT_TARGET_Y_RATIO


def build_reading(
    cctv_id: str,
    edge_id: str | None,
    still_url: str | None,
    reading_core: ReadingCore,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ReadingCore(판독 결과) + 어댑터가 아는 필드 → Reading 전체(dict, 스펙 1장).

    still_url을 "still_public_url" 키로 내보낸다 — backend `cctv_readings`
    테이블 컬럼명(V3__init_cctv_readings.sql)과 정확히 맞춰야 한다.
    """
    return {
        "cctv_id": cctv_id,
        "still_public_url": still_url,
        "edge_id": edge_id,
        "wall_width_m": reading_core.wall_width_m,
        "obstacle_width_m": reading_core.obstacle_width_m,
        "effective_width_m": reading_core.effective_width_m,
        "detected_objects": reading_core.detected_objects,
        "verdict": reading_core.verdict,
        "confidence": reading_core.confidence,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "method": reading_core.method,
        "calibration_error_m": reading_core.calibration_error_m,
        "source_meta": source_meta or {},
    }
