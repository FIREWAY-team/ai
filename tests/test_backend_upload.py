from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import requests
import yaml

from scripts import register_camera
from src.output import backend_upload as upload

KEY = 'uploads/2026-09-17/12345678-1234-1234-1234-123456789abc'
URL = f'https://private.s3.ap-northeast-2.amazonaws.com/{KEY}?X-Amz-Signature=secret'


class Response:
    def __init__(self, status=200, data=None):
        self.status_code, self.data = status, data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def json(self): return self.data


@pytest.fixture
def http(monkeypatch):
    calls = []
    ticket = {'key': KEY, 'upload_url': URL, 'expires_in_seconds': 300}
    def post(url, **kw):
        calls.append(('POST', url, kw))
        return Response(data=deepcopy(ticket))
    def put(url, **kw):
        calls.append(('PUT', url, kw['data'].read(), kw['headers']))
        return Response()
    monkeypatch.setattr(upload.requests, 'post', post)
    monkeypatch.setattr(upload.requests, 'put', put)
    return ticket, calls


@pytest.mark.parametrize('suffix,content_type', [('.png', 'image/png'), ('.mp4', 'video/mp4')])
def test_existing_upload_contract_and_server_key_preserved(http, tmp_path, suffix, content_type):
    _, calls = http
    path = tmp_path / ('source' + suffix)
    path.write_bytes(b'media')
    result = upload.upload_via_backend(str(path), 'https://fireroad.shop/', 'private', 'ap-northeast-2')
    assert calls[0][1] == 'https://fireroad.shop/api/files/upload-url'
    assert calls[0][2]['json'] == {'content_type': content_type}
    assert calls[1] == ('PUT', URL, b'media', {'Content-Type': content_type})
    assert result['key'] == KEY
    assert result['sha256'] == hashlib.sha256(b'media').hexdigest()
    assert 'X-Amz' not in str(result)


@pytest.mark.parametrize('field,value', [('key', 'cctv/abc.mp4'), ('expires_in_seconds', 0),
                                        ('upload_url', 'https://other.test/upload'), ('upload_url', None)])
def test_invalid_ticket_stops_before_put(http, tmp_path, field, value):
    ticket, calls = http
    ticket[field] = value
    path = tmp_path / 'a.png'; path.write_bytes(b'media')
    with pytest.raises(upload.BackendUploadError):
        upload.upload_via_backend(str(path), 'https://fireroad.shop', 'private', 'ap-northeast-2')
    assert len(calls) == 1


@pytest.mark.parametrize('failure', ['rate_limit', 'put_denied', 'timeout'])
def test_failed_upload_preserves_registry_and_hides_signature(http, monkeypatch, tmp_path, failure):
    path = tmp_path / 'a.mp4'; path.write_bytes(b'video')
    registry = tmp_path / 'camera.yaml'
    config = {'cameras': {'camera': {'still_url': str(path), 'wall_width_m': 6.08,
                                    'wall_width_source': 'estimated_avg_of_similar_alleys'}}}
    registry.write_text(yaml.safe_dump(config))
    before = registry.read_bytes()
    if failure == 'rate_limit':
        monkeypatch.setattr(upload.requests, 'post', lambda *a, **kw: Response(429))
    elif failure == 'put_denied':
        monkeypatch.setattr(upload.requests, 'put', lambda *a, **kw: Response(403))
    else:
        def fail(*a, **kw): raise requests.Timeout(URL)
        monkeypatch.setattr(upload.requests, 'put', fail)
    with pytest.raises(upload.BackendUploadError) as error:
        register_camera.register_one(register_camera.CameraEntry('camera', str(path), 6.08),
            'private', 'ap-northeast-2', 'kakao_map', 'low', registry_path=registry,
            upload_api_base_url='https://fireroad.shop')
    assert registry.read_bytes() == before
    assert 'secret' not in str(error.value)


def test_register_uses_existing_api_without_aws_sdk_or_changing_source(http, monkeypatch, tmp_path):
    path = tmp_path / 'a.mp4'; path.write_bytes(b'video')
    registry = tmp_path / 'camera.yaml'
    camera = {'still_url': str(path), 'wall_width_m': 6.08,
              'wall_width_source': 'estimated_avg_of_similar_alleys', 'calibration_source_cctv_id': 'source'}
    registry.write_text(yaml.safe_dump({'cameras': {'camera': camera, 'source': dict(camera)}}))
    monkeypatch.setattr(register_camera, 'upload_media', lambda *a, **kw: pytest.fail('must use existing API'))
    register_camera.register_one(register_camera.CameraEntry('camera', str(path), 6.08),
        'private', 'ap-northeast-2', 'kakao_map', 'low', registry_path=registry,
        upload_api_base_url='https://fireroad.shop')
    actual = yaml.safe_load(registry.read_text())['cameras']['camera']
    assert {k: actual[k] for k in camera} == camera
    assert actual['s3_media']['key'] == KEY


def test_size_policy_checked_before_http(http, tmp_path):
    _, calls = http
    path = tmp_path / 'large.png'
    with path.open('wb') as f: f.truncate(5 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        upload.upload_via_backend(str(path), 'https://fireroad.shop', 'private', 'ap-northeast-2')
    assert not calls
