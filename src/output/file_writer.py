"""출력 — Reading 리스트를 JSON 파일로 저장 (스펙 6-7장, `cctv_readings.json`).

D-9~D-8 단계(L1~L5 판정 결과물)와, 백엔드 연동 전 로컬 확인용으로 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_readings(readings: list[dict[str, Any]], path: str = "cctv_readings.json") -> None:
    """Reading(dict) 리스트를 사람이 읽기 쉬운 JSON으로 저장한다."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(readings, f, ensure_ascii=False, indent=2)
