"""비공개 S3 미디어 참조와 요청 시 발급하는 임시 GET URL."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any
import uuid

_client_cache: dict[tuple[str | None, str | None], Any] = {}


def _get_client(region_name: str | None = None, profile_name: str | None = None) -> Any:
    import boto3
    from botocore.config import Config

    key = (region_name, profile_name)
    if key not in _client_cache:
        session = boto3.Session(profile_name=profile_name)
        _client_cache[key] = session.client("s3", region_name=region_name, config=Config(signature_version="s3v4"))
    return _client_cache[key]


def build_object_key(cctv_id: str, local_path: str, prefix: str = "cctv") -> str:
    return f"{prefix}/{cctv_id}/{uuid.uuid4().hex}{Path(local_path).suffix or '.jpg'}"


def upload_media(
    local_path: str, bucket: str, cctv_id: str, prefix: str = "cctv",
    region_name: str | None = None, *, profile_name: str | None = None,
) -> dict[str, Any]:
    """이미지/영상 업로드. 만료되는 URL 대신 영속 객체 정보만 반환한다."""
    path = Path(local_path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not content_type.startswith(("image/", "video/")):
        raise ValueError("이미지 또는 동영상 파일이 필요합니다")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    size = path.stat().st_size
    if not size:
        raise ValueError("빈 미디어는 업로드할 수 없습니다")
    client = _get_client(region_name, profile_name)
    key = build_object_key(cctv_id, local_path, prefix)
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    return {
        "bucket": bucket, "key": key, "region": region_name or client.meta.region_name,
        "original_filename": path.name, "sha256": digest.hexdigest(),
        "content_type": content_type, "size_bytes": size,
    }


def presign_media(media: dict[str, Any], expires_in: int = 900, *, profile_name: str | None = None) -> str:
    """사용 직전 GET URL 발급. 자격 증명이 먼저 만료되면 URL도 먼저 만료된다."""
    if type(expires_in) is not int or not 1 <= expires_in <= 604800:
        raise ValueError("URL 유효 시간은 1~604800초 정수여야 합니다")
    if not media.get("bucket") or not media.get("key"):
        raise ValueError("S3 bucket과 key가 필요합니다")
    client = _get_client(media.get("region"), profile_name)
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": media["bucket"], "Key": media["key"]},
        ExpiresIn=expires_in, HttpMethod="GET",
    )


def upload_image(
    local_path: str, bucket: str, cctv_id: str, prefix: str = "cctv",
    region_name: str | None = None,
) -> str:
    """기존 호출 호환용 임시 URL. 저장할 때는 upload_media의 객체 참조를 사용한다."""
    return presign_media(upload_media(local_path, bucket, cctv_id, prefix, region_name))
