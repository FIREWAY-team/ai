from __future__ import annotations

from scripts import register_camera
from src import camera_registry


def test_read_manifest_parses_csv(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "cctv_id,image_path,wall_width_m\n"
        "cam_l1,/tmp/l1.jpg,4.2\n"
        "cam_l2,/tmp/l2.jpg,3.05\n",
        encoding="utf-8",
    )

    entries = register_camera.read_manifest(str(manifest))

    assert len(entries) == 2
    assert entries[0] == register_camera.CameraEntry("cam_l1", "/tmp/l1.jpg", 4.2)
    assert entries[1] == register_camera.CameraEntry("cam_l2", "/tmp/l2.jpg", 3.05)


def test_register_one_preserves_original_and_calibration(monkeypatch, tmp_path):
    import yaml
    path = tmp_path / "cameras.yaml"
    original = {"still_url": "/tmp/l1.mp4", "wall_width_m": 6.08,
                "wall_width_source": "estimated_avg_of_similar_alleys", "slope_risk": "unknown",
                "calibration_source_cctv_id": "source"}
    path.write_text(yaml.safe_dump({"cameras": {"cam_l1": original, "source": original}}))
    media = {"bucket": "bucket", "key": "cctv/cam_l1/uuid.mp4", "original_filename": "l1.mp4"}
    calls = []
    def upload(*a, **kw):
        calls.append((a, kw))
        return media
    monkeypatch.setattr(register_camera, "upload_media", upload)
    result = register_camera.register_one(
        register_camera.CameraEntry("cam_l1", "/tmp/l1.mp4", 6.08), "bucket", "ap-northeast-2",
        "kakao_map", "low", profile_name="fireway", registry_path=path,
    )
    assert result == "s3://bucket/cctv/cam_l1/uuid.mp4"
    registered = camera_registry.get_camera("cam_l1", path)
    assert registered == {**original, "s3_media": media}
    assert camera_registry.get_camera("source", path) == original
    assert calls[0][1]["profile_name"] == "fireway"


def test_new_camera_stores_local_source_and_object_reference(monkeypatch, tmp_path):
    path = tmp_path / "cameras.yaml"
    media = {"bucket": "bucket", "key": "key"}
    monkeypatch.setattr(register_camera, "upload_media", lambda *a, **kw: media)
    register_camera.register_one(register_camera.CameraEntry("new", "/tmp/a.png", 5.),
                                 "bucket", "region", "field_measurement", "low", registry_path=path)
    result = camera_registry.get_camera("new", path)
    assert result["still_url"] == "/tmp/a.png"
    assert result["s3_media"] == media


def test_mismatch_rejected_before_upload(monkeypatch, tmp_path):
    import pytest
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera("cam", 5., "/tmp/a.png", path=path)
    before = path.read_bytes()
    monkeypatch.setattr(register_camera, "upload_media", lambda *a, **kw: pytest.fail("must not upload"))
    for file, width in [("/tmp/other.png", 5.), ("/tmp/a.png", 9.)]:
        with pytest.raises(ValueError):
            register_camera.register_one(register_camera.CameraEntry("cam", file, width),
                                        "bucket", "region", "kakao_map", "low", registry_path=path)
    assert path.read_bytes() == before


def test_upload_failure_preserves_registry(monkeypatch, tmp_path):
    import pytest
    path = tmp_path / "cameras.yaml"
    camera_registry.register_camera("cam", 5., "/tmp/a.png", path=path)
    before = path.read_bytes()
    def fail(*a, **kw):
        raise OSError("network failure")
    monkeypatch.setattr(register_camera, "upload_media", fail)
    with pytest.raises(OSError):
        register_camera.register_one(register_camera.CameraEntry("cam", "/tmp/a.png", 5.),
                                    "bucket", "region", "kakao_map", "low", registry_path=path)
    assert path.read_bytes() == before
