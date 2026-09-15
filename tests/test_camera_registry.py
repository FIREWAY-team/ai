from __future__ import annotations

import pytest

from src import camera_registry


def test_load_cameras_empty_when_file_missing(tmp_path):
    missing = tmp_path / "cameras.yaml"
    assert camera_registry.load_cameras(missing) == {}


def test_register_and_get_camera_round_trips(tmp_path):
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera(
        cctv_id="cam_l1",
        wall_width_m=4.2,
        still_url="https://bucket.s3.amazonaws.com/cctv/cam_l1/abc.jpg",
        path=path,
    )

    camera = camera_registry.get_camera("cam_l1", path=path)
    assert camera["wall_width_m"] == 4.2
    assert camera["still_url"] == "https://bucket.s3.amazonaws.com/cctv/cam_l1/abc.jpg"
    assert camera["wall_width_source"] == "kakao_map"
    assert camera["slope_risk"] == "low"


def test_get_camera_raises_for_unregistered_cctv_id(tmp_path):
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera("cam_l1", 4.2, "https://example.com/a.jpg", path=path)

    with pytest.raises(KeyError):
        camera_registry.get_camera("cam_unknown", path=path)


def test_register_camera_upserts_without_dropping_other_cameras(tmp_path):
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera("cam_l1", 4.2, "https://example.com/a.jpg", path=path)
    camera_registry.register_camera("cam_l2", 3.05, "https://example.com/b.jpg", path=path)

    cameras = camera_registry.load_cameras(path)
    assert set(cameras.keys()) == {"cam_l1", "cam_l2"}
    assert cameras["cam_l1"]["wall_width_m"] == 4.2
    assert cameras["cam_l2"]["wall_width_m"] == 3.05


def test_register_camera_overwrites_existing_entry(tmp_path):
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera("cam_l1", 4.2, "https://example.com/old.jpg", path=path)
    camera_registry.register_camera("cam_l1", 4.5, "https://example.com/new.jpg", path=path)

    camera = camera_registry.get_camera("cam_l1", path=path)
    assert camera["wall_width_m"] == 4.5
    assert camera["still_url"] == "https://example.com/new.jpg"


@pytest.mark.parametrize("source", [
    None,
    {"still_url": "other.mp4"},
    {"still_url": "clip.mp4", "calibration_source_cctv_id": "alias"},
])
def test_calibration_alias_rejects_missing_mismatched_or_nested_source(monkeypatch, source):
    cameras = {"alias": {"still_url": "clip.mp4", "calibration_source_cctv_id": "original"}}
    if source is not None:
        cameras["original"] = source
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: cameras)
    with pytest.raises(ValueError):
        camera_registry.calibration_source_id("alias")
