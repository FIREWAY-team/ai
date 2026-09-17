"""기존 POST /api/files/upload-url → 서명된 PUT 계약에 맞춘 미디어 업로드."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import requests


class BackendUploadError(RuntimeError):
    pass


def upload_via_backend(local_path: str, base_url: str, bucket: str, region: str) -> dict:
    """서버 발급 key 보존. 신고 첨부 생성이나 AI 판정 전송은 수행하지 않는다."""
    path = Path(local_path)
    content_type = mimetypes.guess_type(path.name)[0]
    allowed = {'image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/quicktime'}
    if content_type not in allowed:
        raise ValueError('기존 백엔드에서 허용하지 않는 미디어 유형입니다')
    size = path.stat().st_size
    limit = 50 * 1024 * 1024 if content_type.startswith('video/') else 5 * 1024 * 1024
    if not 0 < size <= limit:
        raise ValueError('기존 업로드 크기 제한을 벗어났습니다 (사진 5MiB, 영상 50MiB)')
    parsed_base = urlsplit(base_url)
    if parsed_base.scheme != 'https' or not parsed_base.netloc or parsed_base.query or parsed_base.fragment:
        raise ValueError('백엔드 HTTPS 기본 주소가 필요합니다')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    try:
        with requests.post(base_url.rstrip('/') + '/api/files/upload-url',
                           json={'content_type': content_type}, timeout=(10, 30), allow_redirects=False) as response:
            if response.status_code != 200:
                raise BackendUploadError(f'업로드 URL 발급 실패 (HTTP {response.status_code})')
            try:
                ticket = response.json()
            except ValueError:
                raise BackendUploadError('업로드 URL 응답이 JSON이 아닙니다') from None
        if not isinstance(ticket, dict):
            raise BackendUploadError('업로드 URL 응답 형식 불일치')
        key, url, ttl = ticket.get('key'), ticket.get('upload_url'), ticket.get('expires_in_seconds')
        if not isinstance(key, str) or not re.fullmatch(r'uploads/\d{4}-\d{2}-\d{2}/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', key):
            raise BackendUploadError('백엔드 발급 key 형식 불일치')
        if not isinstance(url, str) or type(ttl) is not int or ttl <= 0:
            raise BackendUploadError('업로드 URL 또는 만료 정보 누락')
        parsed = urlsplit(url)
        expected_hosts = {f'{bucket}.s3.{region}.amazonaws.com', f'{bucket}.s3.amazonaws.com'}
        if parsed.scheme != 'https' or parsed.netloc not in expected_hosts or unquote(parsed.path) != '/' + key:
            raise BackendUploadError('서명 URL의 버킷 또는 객체 key가 설정과 다릅니다')
        with path.open('rb') as source:
            with requests.put(url, data=source, headers={'Content-Type': content_type},
                              timeout=(10, 120), allow_redirects=False) as response:
                if response.status_code not in {200, 204}:
                    raise BackendUploadError(f'서명된 미디어 PUT 실패 (HTTP {response.status_code})')
    except requests.RequestException:
        raise BackendUploadError('미디어 업로드 네트워크 오류 또는 시간 초과') from None
    return {'bucket': bucket, 'key': key, 'region': region, 'original_filename': path.name,
            'sha256': digest.hexdigest(), 'content_type': content_type, 'size_bytes': size}
