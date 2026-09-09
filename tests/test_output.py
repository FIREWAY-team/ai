from __future__ import annotations

import json

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
    def __init__(self, status_code: int = 200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_post_reading_posts_to_readings_endpoint(monkeypatch):
    captured = {}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(api_client.requests, "post", _fake_post)

    response = api_client.post_reading("https://backend.internal", SAMPLE_READING)

    assert captured["url"] == "https://backend.internal/readings"
    assert captured["json"] == SAMPLE_READING
    assert response.status_code == 200


def test_post_readings_sends_each_reading(monkeypatch):
    posted = []
    monkeypatch.setattr(
        api_client, "post_reading", lambda base_url, reading, timeout_sec=10: posted.append(reading)
    )

    api_client.post_readings("https://backend.internal", [SAMPLE_READING, SAMPLE_READING])

    assert len(posted) == 2
