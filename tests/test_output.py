from __future__ import annotations

import json

import requests

from src.output import api_client, file_writer

SAMPLE_READING = {
    "cctv_id": "cam_l1",
    "verdict": {"pump-3.5": "PASS", "pump-8": "UNCERTAIN"},
    "confidence": 0.7,
}


def test_write_readings_round_trips_json(tmp_path):
    out_path = tmp_path / "nested" / "cctv_readings.json"
    file_writer.write_readings([SAMPLE_READING], path=str(out_path))

    with open(out_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved == [SAMPLE_READING]


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


def test_post_reading_posts_to_readings_endpoint(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    result = api_client.post_reading("https://backend.internal", SAMPLE_READING)

    assert captured["url"] == "https://backend.internal/readings"
    assert captured["json"] == SAMPLE_READING
    assert result == {"ok": True, "status_code": 200, "cctv_id": "cam_l1"}


def test_post_reading_connection_error_returns_error_dict(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("연결 거부")

    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    result = api_client.post_reading("https://backend.internal", SAMPLE_READING)

    assert result["ok"] is False
    assert result["cctv_id"] == "cam_l1"
    assert "연결" in result["error"]


def test_post_reading_timeout_returns_error_dict(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("시간 초과")

    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    result = api_client.post_reading("https://backend.internal", SAMPLE_READING, timeout_sec=5)

    assert result["ok"] is False
    assert "5초" in result["error"]


def test_post_reading_http_error_returns_error_dict(monkeypatch):
    def _fake_post(url, json=None, timeout=None):
        return _FakeResponse(500, text="internal error")

    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    result = api_client.post_reading("https://backend.internal", SAMPLE_READING)

    assert result["ok"] is False
    assert "500" in result["error"]
    assert result["detail"] == "internal error"


def test_post_readings_continues_after_one_failure(monkeypatch):
    calls = {"n": 0}

    def _fake_post_reading(base_url, reading, timeout_sec=10):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": False, "error": "backend 서버에 연결할 수 없습니다", "cctv_id": reading["cctv_id"]}
        return {"ok": True, "status_code": 200, "cctv_id": reading["cctv_id"]}

    monkeypatch.setattr(api_client, "post_reading", _fake_post_reading)

    results = api_client.post_readings("https://backend.internal", [SAMPLE_READING, SAMPLE_READING])

    assert len(results) == 2
    assert results[0]["ok"] is False
    assert results[1]["ok"] is True  # 첫 건이 실패해도 두 번째 건은 계속 시도된다
