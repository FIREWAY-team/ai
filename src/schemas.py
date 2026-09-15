"""판독 모듈이 채우는 팀 공용 Reading 스키마의 하위 집합 (스펙 1장, v4)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReadingCore:
    """Reading의 하위 집합 — 판독 모듈이 책임지는 필드만.

    나머지 필드(cctv_id, still_url, edge_id, measured_at, source_meta)는
    어댑터(src/adapters/*) 쪽에서 채워 감싼다.
    """

    wall_width_m: float
    obstacle_width_m: float
    effective_width_m: float
    detected_objects: list[dict] = field(default_factory=list)
    verdict: dict[str, str] = field(default_factory=dict)  # {"pump-3.5": "PASS", ...}
    confidence: float = 0.0  # 차종별 판정 확률 중 가장 보수적인(작은) 값
    calibration_error_m: float = 0.0  # 이 프레임 캘리브레이션 추정 오차
    method: str = "yolov11_homography_v5"
    quality_flags: list[str] = field(default_factory=list)
