"""출력 — CCTV 정지 이미지를 S3에 업로드해 still_url을 만든다.

CCTV 12곳은 위치·화면이 고정된 데모용 자산이라 이미지 자체가 계속 쌓이는
동적 스트림은 아니지만, 팀 Reading 스키마(still_url)가 실제 URL 참조를
전제로 하므로 정적 호스팅용 S3에 올린다(프리티어 5GB로 충분 — 이미지
12장 + 모션 데모 10장 정도는 몇 MB 수준). 실측값(wall_width_m)은 이미지가
아니라 cctv_id(카메라)에 묶는다 — src.camera_registry 참고.

파일명은 UUID로 만든다: 같은 카메라를 여러 번 재등록해도(예: 재촬영, 모션
데모 프레임 10장) 기존 객체를 덮어쓰지 않기 위해서다.
"""
from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Any

_client_cache: dict[str, Any] = {}


def _get_client(region_name: str | None = None) -> Any:
    import boto3  # 무거운 의존성은 아니지만 실제 업로드 시점에만 필요

    key = region_name or "default"
    if key not in _client_cache:
        _client_cache[key] = boto3.client("s3", region_name=region_name)
    return _client_cache[key]


def build_object_key(cctv_id: str, local_path: str, prefix: str = "cctv") -> str:
    """cctv_id를 경로에 남겨 버킷만 봐도 어느 카메라 것인지 알 수 있게 하되,
    파일명 자체는 충돌 방지를 위해 UUID로 만든다."""
    ext = Path(local_path).suffix or ".jpg"
    return f"{prefix}/{cctv_id}/{uuid.uuid4().hex}{ext}"


def upload_image(
    local_path: str,
    bucket: str,
    cctv_id: str,
    prefix: str = "cctv",
    region_name: str | None = None,
) -> str:
    """이미지 한 장을 S3에 업로드하고 공개 정적 호스팅 URL을 반환한다.

    버킷은 정적 웹사이트 호스팅(또는 public-read 버킷 정책)이 이미 켜져
    있다고 가정한다 — 이 함수는 버킷을 만들거나 정책을 바꾸지 않는다
    (인프라 프로비저닝은 scripts/setup_s3_bucket.py 참고).
    """
    key = build_object_key(cctv_id, local_path, prefix)
    content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"

    client = _get_client(region_name)
    client.upload_file(
        local_path,
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )

    region = region_name or client.meta.region_name
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
