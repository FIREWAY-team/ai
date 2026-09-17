from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.inference import calibration_store
from src import camera_registry


def test_load_observations_empty_when_never_accumulated(tmp_path):
    result = calibration_store.load_observations("cctv_never_seen", calibration_dir=tmp_path)
    assert result == []


def test_append_then_load_round_trip(tmp_path):
    obs = [
        calibration_store.make_observation(
            scale=0.0113, confidence=0.6, footpoint_y=1024.9, camera_height_px=1080.0
        ),
        calibration_store.make_observation(
            scale=0.0120, confidence=0.8, footpoint_y=500.0, camera_height_px=1080.0
        ),
    ]
    calibration_store.append_observations("cctv_1", obs, calibration_dir=tmp_path)

    loaded = calibration_store.load_observations("cctv_1", calibration_dir=tmp_path)
    assert len(loaded) == 2
    assert loaded[0].scale == 0.0113
    assert loaded[1].footpoint_y == 500.0


def test_append_accumulates_across_multiple_calls(tmp_path):
    # 매 실행(스크립트 주기 실행)마다 append만 하고 기존 걸 안 지운다 — 이게
    # 이 저장소의 핵심 목적(여러 시점에 걸쳐 누적).
    first = [calibration_store.make_observation(0.01, 0.9, 100.0, 1080.0)]
    second = [calibration_store.make_observation(0.011, 0.8, 200.0, 1080.0)]
    calibration_store.append_observations("cctv_history_test", first, calibration_dir=tmp_path)
    calibration_store.append_observations("cctv_history_test", second, calibration_dir=tmp_path)

    loaded = calibration_store.load_observations("cctv_history_test", calibration_dir=tmp_path)
    assert len(loaded) == 2


def test_append_empty_list_creates_no_file(tmp_path):
    calibration_store.append_observations("cctv_empty_history_test", [], calibration_dir=tmp_path)
    assert calibration_store.load_observations("cctv_empty_history_test", calibration_dir=tmp_path) == []
    assert not list(tmp_path.rglob("*.jsonl"))


def test_load_observations_filters_by_max_age_days(tmp_path):
    old_obs = calibration_store.CalibrationObservation(
        scale=0.01,
        confidence=0.9,
        footpoint_y=100.0,
        camera_height_px=1080.0,
        observed_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
    )
    recent_obs = calibration_store.make_observation(0.011, 0.8, 200.0, 1080.0)
    calibration_store.append_observations("cctv_4", [old_obs, recent_obs], calibration_dir=tmp_path)

    all_loaded = calibration_store.load_observations("cctv_4", calibration_dir=tmp_path)
    assert len(all_loaded) == 2

    recent_only = calibration_store.load_observations(
        "cctv_4", max_age_days=7, calibration_dir=tmp_path
    )
    assert len(recent_only) == 1
    assert recent_only[0].footpoint_y == 200.0


def test_make_observation_casts_numpy_float32_to_python_float(tmp_path):
    # cv2.boxPoints() 등 OpenCV 연산 결과가 그대로 들어오면 numpy.float32라
    # json.dumps가 실패한다(2026-09-15 accumulate_calibration.py 실행 중 발견:
    # "Object of type float32 is not JSON serializable"). make_observation()
    # 시점에 파이썬 float로 캐스팅해서 저장/로드가 끝까지 되는지 확인한다.
    obs = calibration_store.make_observation(
        scale=np.float32(0.0113),
        confidence=np.float32(0.6),
        footpoint_y=np.float32(1024.9),
        camera_height_px=np.float32(1080.0),
    )
    assert type(obs.scale) is float
    assert type(obs.footpoint_y) is float

    calibration_store.append_observations("cctv_np", [obs], calibration_dir=tmp_path)
    loaded = calibration_store.load_observations("cctv_np", calibration_dir=tmp_path)
    assert len(loaded) == 1
    assert abs(loaded[0].scale - 0.0113) < 1e-6


def test_different_cameras_stored_separately(tmp_path):
    calibration_store.append_observations(
        "cctv_a", [calibration_store.make_observation(0.01, 0.9, 100.0, 1080.0)], calibration_dir=tmp_path
    )
    calibration_store.append_observations(
        "cctv_b", [calibration_store.make_observation(0.02, 0.9, 100.0, 1080.0)], calibration_dir=tmp_path
    )
    assert len(calibration_store.load_observations("cctv_a", calibration_dir=tmp_path)) == 1
    assert len(calibration_store.load_observations("cctv_b", calibration_dir=tmp_path)) == 1


def test_alias_reads_and_appends_to_original_without_copying_observations(monkeypatch, tmp_path):
    cameras = {
        "original": {"still_url": "clip.mp4"},
        "alias": {"still_url": "clip.mp4", "calibration_source_cctv_id": "original"},
    }
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: cameras)
    first = calibration_store.make_observation(.01, .9, 100., 720.)
    second = calibration_store.make_observation(.02, .9, 200., 720.)
    calibration_store.append_observations("original", [first], calibration_dir=tmp_path)
    assert calibration_store.load_observations("alias", calibration_dir=tmp_path) == [first]
    calibration_store.append_observations("alias", [second], calibration_dir=tmp_path)
    assert calibration_store.load_observations("original", calibration_dir=tmp_path) == [first, second]
    assert (tmp_path / calibration_store.SCALE_VERSION / "original.jsonl").exists()
    assert not (tmp_path / calibration_store.SCALE_VERSION / "alias.jsonl").exists()
