"""출력 — 박종준 백엔드로 Reading POST (스펙 7장, D-7 Gate 1 대응).

엔드포인트/인증 방식은 백엔드 확정 전까지 미정이므로 base_url을 호출부가
주입한다. Reading 스키마(dict) 자체는 어댑터(src/adapters/common.py)가
이미 팀 확정 필드명으로 조립해 넘기므로 이 모듈은 전송만 담당한다.

연결 실패/타임아웃/HTTP 에러를 예외로 던지지 않고 {"ok": False, "error": ...}
형태의 데이터로 반환한다 — 12개 카메라를 순차 전송하는 post_readings()에서
한 건이 실패해도(backend가 잠깐 불안정한 경우 등) 예외가 전체를 멈추지
않고 나머지 판정을 계속 전송할 수 있게 하기 위함.
"""
from __future__ import annotations

from typing import Any

import requests

REQUEST_TIMEOUT_SEC = 10


def post_reading(
    base_url: str, reading: dict[str, Any], timeout_sec: int = REQUEST_TIMEOUT_SEC
) -> dict[str, Any]:
    """Reading 한 건을 백엔드 `/readings` 엔드포인트로 POST한다.

    성공: {"ok": True, "status_code": ..., "cctv_id": ...}
    실패: {"ok": False, "error": "...", "cctv_id": ...} (연결 실패/타임아웃/HTTP 에러 공통)
    """
    cctv_id = reading.get("cctv_id")
    try:
        response = requests.post(f"{base_url}/readings", json=reading, timeout=timeout_sec)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "backend 서버에 연결할 수 없습니다", "cctv_id": cctv_id}
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": f"backend 응답이 {timeout_sec}초 내에 오지 않았습니다",
            "cctv_id": cctv_id,
        }
    except requests.exceptions.HTTPError as error:
        return {
            "ok": False,
            "error": f"backend가 오류를 반환했습니다 (HTTP {error.response.status_code})",
            "detail": error.response.text[:200],
            "cctv_id": cctv_id,
        }
    return {"ok": True, "status_code": response.status_code, "cctv_id": cctv_id}


def post_readings(
    base_url: str, readings: list[dict[str, Any]], timeout_sec: int = REQUEST_TIMEOUT_SEC
) -> list[dict[str, Any]]:
    """Reading 여러 건을 순차 POST한다. 한 건이 실패해도 예외가 전파되지
    않으므로 나머지 건은 계속 전송된다 — 각 결과의 "ok" 필드로 성공 여부를
    판단한다.
    """
    return [post_reading(base_url, reading, timeout_sec) for reading in readings]
