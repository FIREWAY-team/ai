from __future__ import annotations

import json

from src.output import file_writer

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
