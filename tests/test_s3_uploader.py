from __future__ import annotations

from src.output import s3_uploader


class _FakeS3Client:
    def __init__(self, region_name: str):
        self.meta = type("Meta", (), {"region_name": region_name})()
        self.uploaded = []

    def upload_file(self, local_path, bucket, key, ExtraArgs=None):
        self.uploaded.append((local_path, bucket, key, ExtraArgs))


def test_build_object_key_includes_cctv_id_and_uuid():
    key = s3_uploader.build_object_key("cam_l1", "/tmp/photo.jpg")
    assert key.startswith("cctv/cam_l1/")
    assert key.endswith(".jpg")
    # UUID(hex, 32자) + 확장자 형태인지
    filename = key.rsplit("/", 1)[-1]
    assert len(filename) == len("00000000000000000000000000000000.jpg")


def test_build_object_key_is_unique_per_call():
    key1 = s3_uploader.build_object_key("cam_l1", "/tmp/photo.jpg")
    key2 = s3_uploader.build_object_key("cam_l1", "/tmp/photo.jpg")
    assert key1 != key2  # 재등록해도 기존 객체를 덮어쓰지 않아야 한다


def test_upload_image_returns_regional_url(monkeypatch, tmp_path):
    fake_client = _FakeS3Client(region_name="ap-northeast-2")
    monkeypatch.setattr(s3_uploader, "_get_client", lambda region_name=None: fake_client)

    image = tmp_path / "cam_l1.jpg"
    image.write_bytes(b"fake-jpeg-bytes")

    url = s3_uploader.upload_image(str(image), bucket="fireway-cctv-demo", cctv_id="cam_l1")

    assert url.startswith("https://fireway-cctv-demo.s3.ap-northeast-2.amazonaws.com/cctv/cam_l1/")
    assert len(fake_client.uploaded) == 1
    local_path, bucket, key, extra_args = fake_client.uploaded[0]
    assert local_path == str(image)
    assert bucket == "fireway-cctv-demo"
    assert extra_args == {"ContentType": "image/jpeg"}


def test_upload_image_us_east_1_uses_bare_url(monkeypatch, tmp_path):
    fake_client = _FakeS3Client(region_name="us-east-1")
    monkeypatch.setattr(s3_uploader, "_get_client", lambda region_name=None: fake_client)

    image = tmp_path / "cam_l2.png"
    image.write_bytes(b"fake-png-bytes")

    url = s3_uploader.upload_image(str(image), bucket="fireway-cctv-demo", cctv_id="cam_l2")

    assert url.startswith("https://fireway-cctv-demo.s3.amazonaws.com/cctv/cam_l2/")
