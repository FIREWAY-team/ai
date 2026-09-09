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


def test_register_one_uploads_and_writes_registry(monkeypatch):
    # camera_registry.register_camera 자체를 monkeypatch해서, 실제
    # configs/cameras.yaml을 건드리지 않고 register_one이 넘기는 인자만 검증한다.
    monkeypatch.setattr(
        register_camera,
        "upload_image",
        lambda local_path, bucket, cctv_id, region_name=None: f"https://{bucket}.s3.amazonaws.com/{cctv_id}.jpg",
    )
    calls = []
    monkeypatch.setattr(
        camera_registry, "register_camera", lambda **kwargs: calls.append(kwargs)
    )

    entry = register_camera.CameraEntry("cam_l1", "/tmp/l1.jpg", 4.2)
    still_url = register_camera.register_one(
        entry, bucket="fireway-cctv-demo", region="ap-northeast-2",
        wall_width_source="kakao_map", slope_risk="low",
    )

    assert still_url == "https://fireway-cctv-demo.s3.amazonaws.com/cam_l1.jpg"
    assert len(calls) == 1
    assert calls[0]["cctv_id"] == "cam_l1"
    assert calls[0]["wall_width_m"] == 4.2
    assert calls[0]["still_url"] == still_url
