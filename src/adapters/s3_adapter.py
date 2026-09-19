"""비공개 S3 원본을 임시 파일로 검증한 뒤 기존 디코더에 전달한다."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

import requests

from src.output.s3_uploader import presign_media


class MediaDownloadError(RuntimeError):
    """일시적인 전송 실패. 차량 통행 판정으로 변환하지 않는다."""


def validate_media(media: dict, original_path: str, adapter: str) -> None:
    expected_type = {"file": "image/", "video": "video/"}.get(adapter)
    if not expected_type or not str(media.get("content_type", "")).startswith(expected_type):
        raise ValueError("등록 원본과 S3 미디어 유형이 다릅니다")
    filename = Path(unquote(urlsplit(original_path).path)).name
    if media.get("original_filename") != filename:
        raise ValueError("등록 원본과 S3 원본 파일명이 다릅니다")
    if not media.get("bucket") or not media.get("key"):
        raise ValueError("S3 객체 참조가 없습니다")
    if not re.fullmatch(r"[0-9a-f]{64}", str(media.get("sha256", ""))):
        raise ValueError("S3 원본 SHA-256이 필요합니다")
    if type(media.get("size_bytes")) is not int or media["size_bytes"] <= 0:
        raise ValueError("S3 원본 크기는 양수 정수여야 합니다")


@contextmanager
def downloaded_media(media: dict, original_path: str, adapter: str, *, profile_name: str | None = None):
    """서명 URL은 메모리에서만 사용. 403은 한 번 재발급, 실패 시 임시 파일 삭제."""
    validate_media(media, original_path, adapter)
    with TemporaryDirectory(prefix="fireway-media-") as directory:
        path = Path(directory) / ("source" + Path(media["original_filename"]).suffix)
        for attempt in range(2):
            try:
                url = presign_media(media, profile_name=profile_name)
            except Exception:
                # SDK 오류의 인증 정보를 사용자용 예외에 포함하지 않는다.
                raise MediaDownloadError("S3 읽기 URL 발급 실패") from None
            try:
                with requests.get(url, stream=True, timeout=(10, 60), allow_redirects=False) as response:
                    if response.status_code == 403 and attempt == 0:
                        continue
                    if response.status_code != 200:
                        raise MediaDownloadError(f"S3 미디어 다운로드 실패 (HTTP {response.status_code})")
                    digest = hashlib.sha256()
                    size = 0
                    with path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            size += len(chunk)
                            if size > media["size_bytes"]:
                                raise ValueError("S3 미디어 크기가 등록 원본과 다릅니다")
                            digest.update(chunk)
                            output.write(chunk)
                    if size != media["size_bytes"] or digest.hexdigest() != media["sha256"]:
                        raise ValueError("S3 미디어 크기 또는 SHA-256이 등록 원본과 다릅니다")
                    break
            except requests.RequestException:
                raise MediaDownloadError("S3 미디어 전송 실패 또는 시간 초과") from None
        yield str(path)
