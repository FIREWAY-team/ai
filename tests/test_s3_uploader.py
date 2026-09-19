import hashlib

import pytest

from src.output import s3_uploader as s3


class FakeClient:
    meta = type('Meta', (), {'region_name': 'ap-northeast-2'})()

    def __init__(self):
        self.uploads = []
        self.signatures = []

    def upload_file(self, *args, **kwargs):
        self.uploads.append((args, kwargs))

    def generate_presigned_url(self, *args, **kwargs):
        self.signatures.append((args, kwargs))
        return f'https://example.test/private?X-Amz-Signature=temporary-{len(self.signatures)}'


@pytest.mark.parametrize('filename,content_type', [('frame.png', 'image/png'), ('clip.mp4', 'video/mp4')])
def test_private_upload_and_renewable_get_url(monkeypatch, tmp_path, filename, content_type):
    client = FakeClient()
    calls = []
    def get_client(region=None, profile=None):
        calls.append((region, profile))
        return client
    monkeypatch.setattr(s3, '_get_client', get_client)
    path = tmp_path / filename
    path.write_bytes(b'media')
    media = s3.upload_media(str(path), 'bucket', 'camera', region_name='ap-northeast-2', profile_name='fireway')
    assert media['sha256'] == hashlib.sha256(b'media').hexdigest()
    assert media['original_filename'] == filename
    assert media['size_bytes'] == 5
    assert 'url' not in media and 'X-Amz' not in str(media)
    assert client.uploads[0][1] == {'ExtraArgs': {'ContentType': content_type}}
    assert not client.signatures
    first = s3.presign_media(media, profile_name='fireway')
    second = s3.presign_media(media, expires_in=60, profile_name='fireway')
    assert first != second
    assert client.signatures[-1] == (('get_object',), {
        'Params': {'Bucket': 'bucket', 'Key': media['key']}, 'ExpiresIn': 60, 'HttpMethod': 'GET'})
    assert all(call == ('ap-northeast-2', 'fireway') for call in calls)


@pytest.mark.parametrize('ttl', [0, -1, 604801, 1.5, True])
def test_invalid_ttl_rejected_before_client_creation(monkeypatch, ttl):
    monkeypatch.setattr(s3, '_get_client', lambda *a: pytest.fail('must not create client'))
    with pytest.raises(ValueError):
        s3.presign_media({'bucket': 'b', 'key': 'k'}, ttl)


def test_unique_keys():
    assert s3.build_object_key('camera', 'a.mp4') != s3.build_object_key('camera', 'a.mp4')


def test_failed_upload_does_not_return_reference(monkeypatch, tmp_path):
    client = FakeClient()
    def fail(*a, **kw):
        raise OSError('upload failed')
    client.upload_file = fail
    monkeypatch.setattr(s3, '_get_client', lambda *a: client)
    path = tmp_path / 'a.mp4'
    path.write_bytes(b'video')
    with pytest.raises(OSError):
        s3.upload_media(str(path), 'bucket', 'camera')
    assert not client.signatures


def test_legacy_wrapper_returns_signed_url(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(s3, '_get_client', lambda *a: client)
    path = tmp_path / 'a.png'
    path.write_bytes(b'image')
    assert 'X-Amz-Signature=' in s3.upload_image(str(path), 'bucket', 'camera')


def test_real_sdk_signs_get_offline(monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    import boto3
    from botocore.config import Config
    client = boto3.client('s3', region_name='ap-northeast-2',
                          aws_access_key_id='test-only', aws_secret_access_key='test-only',
                          config=Config(signature_version='s3v4'))
    monkeypatch.setattr(s3, '_get_client', lambda *a: client)
    url = s3.presign_media({'bucket': 'test-bucket', 'key': 'cctv/clip.mp4', 'region': 'ap-northeast-2'}, 900)
    query = parse_qs(urlsplit(url).query)
    assert query['X-Amz-Algorithm'] == ['AWS4-HMAC-SHA256']
    assert query['X-Amz-Expires'] == ['900']
    assert urlsplit(url).path.endswith('/cctv/clip.mp4')
