"""출력 — 박종준 백엔드로 Reading POST (스펙 7장, D-7 Gate 1 대응).

엔드포인트/인증 방식은 백엔드 확정 전까지 미정이므로 base_url을 호출부가
주입한다. Reading 스키마(dict) 자체는 어댑터(src/adapters/common.py)가
이미 팀 확정 필드명으로 조립해 넘기므로 이 모듈은 전송만 담당한다.
"""
from __future__ import annotations

from typing import Any

import requests

REQUEST_TIMEOUT_SEC = 10


def post_reading(
    base_url: str, reading: dict[str, Any], timeout_sec: int = REQUEST_TIMEOUT_SEC
) -> requests.Response:
    """Reading 한 건을 백엔드 `/readings` 엔드포인트로 POST한다."""
    response = requests.post(f"{base_url}/readings", json=reading, timeout=timeout_sec)
    response.raise_for_status()
    return response


def post_readings(
    base_url: str, readings: list[dict[str, Any]], timeout_sec: int = REQUEST_TIMEOUT_SEC
) -> list[requests.Response]:
    """Reading 여러 건을 순차 POST한다. 실패한 건은 예외가 그대로 전파되므로
    호출부가 부분 실패를 알 수 있다(어느 항목에서 멈췄는지는 예외 이전
    반환 리스트 길이로 파악 가능).
    """
    return [post_reading(base_url, reading, timeout_sec) for reading in readings]
