import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from scripts import refresh_moran_readings as refresh
from src.adapters import s3_adapter as adapter
from src.schemas import ReadingCore, MeasurementUnavailableError


class Response:
    def __init__(self, content=b'', status=200):
        self.content, self.status_code = content, status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, chunk_size):
        yield self.content


def media_for(path):
    content = path.read_bytes()
    return {'bucket': 'private', 'key': 'uuid' + path.suffix, 'region': 'ap-northeast-2',
            'content_type': 'video/mp4' if path.suffix == '.mp4' else 'image/png',
            'original_filename': path.name, 'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}


@pytest.fixture
def scene(tmp_path, monkeypatch):
    path = tmp_path / 'image.png'
    cv2.imwrite(str(path), np.full((32, 48, 3), 128, np.uint8))
    media = media_for(path)
    monkeypatch.setattr(adapter, 'presign_media', lambda *a, **kw: 'https://example.test/?X-Amz-Signature=secret')
    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **kw: Response(path.read_bytes()))
    return path, media


def test_download_cleanup_on_success_and_consumer_failure(scene):
    original, media = scene
    with adapter.downloaded_media(media, str(original), 'file') as local:
        assert Path(local).read_bytes() == original.read_bytes()
    assert not Path(local).exists()
    with pytest.raises(RuntimeError):
        with adapter.downloaded_media(media, str(original), 'file') as local:
            raise RuntimeError('decoder failed')
    assert not Path(local).exists()


@pytest.mark.parametrize('content', [b'', b'corrupt', b'x' * 10000])
def test_size_and_hash_mismatch_rejected(scene, monkeypatch, content):
    original, media = scene
    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **kw: Response(content))
    with pytest.raises(ValueError):
        with adapter.downloaded_media(media, str(original), 'file'):
            pytest.fail('must not decode')


def test_same_size_wrong_hash_rejected(scene, monkeypatch):
    original, media = scene
    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **kw: Response(b'x' * media['size_bytes']))
    with pytest.raises(ValueError, match='SHA-256'):
        with adapter.downloaded_media(media, str(original), 'file'):
            pytest.fail('must not decode')


@pytest.mark.parametrize('statuses, succeeds', [([403, 200], True), ([403, 403], False), ([404], False), ([302], False)])
def test_url_renewal_is_bounded(scene, monkeypatch, statuses, succeeds):
    original, media = scene
    responses = [Response(original.read_bytes(), status) for status in statuses]
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return responses[len(calls) - 1]
    monkeypatch.setattr(adapter.requests, 'get', get)
    if succeeds:
        with adapter.downloaded_media(media, str(original), 'file') as local:
            assert Path(local).exists()
    else:
        with pytest.raises(adapter.MediaDownloadError) as error:
            with adapter.downloaded_media(media, str(original), 'file'):
                pytest.fail('must not decode')
        assert 'secret' not in str(error.value)
    assert len(calls) == len(statuses)
    assert all(r.closed for r in responses)


def test_timeout_does_not_expose_signed_url(scene, monkeypatch):
    original, media = scene
    def fail(*a, **kw):
        raise requests.Timeout('https://example.test/?X-Amz-Signature=secret')
    monkeypatch.setattr(adapter.requests, 'get', fail)
    with pytest.raises(adapter.MediaDownloadError) as error:
        with adapter.downloaded_media(media, str(original), 'file'):
            pytest.fail('must not decode')
    assert 'secret' not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize('field,value', [('original_filename', 'other.png'), ('sha256', 'invalid'),
                                       ('size_bytes', 0), ('content_type', 'video/mp4'), ('key', '')])
def test_invalid_reference_rejected_before_network(scene, monkeypatch, field, value):
    original, media = scene
    media[field] = value
    monkeypatch.setattr(adapter, 'presign_media', lambda *a, **kw: pytest.fail('no network'))
    with pytest.raises(ValueError):
        with adapter.downloaded_media(media, str(original), 'file'):
            pytest.fail('must not decode')


@pytest.mark.parametrize('video', [False, True])
def test_refresh_local_and_s3_use_same_frames_policy_and_result(scene, monkeypatch, video):
    original, _ = scene
    if video:
        original = original.with_suffix('.mp4')
        writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*'mp4v'), 5, (48, 32))
        assert writer.isOpened()
        for i in range(50):
            writer.write(np.full((32, 48, 3), i * 4, np.uint8))
        writer.release()
    media = media_for(original)
    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **kw: Response(original.read_bytes()))
    camera = {'still_url': str(original), 'wall_width_m': 6.08, 's3_media': media}
    monkeypatch.setattr(refresh.camera_registry, 'get_camera', lambda _: camera)
    monkeypatch.setattr(refresh.camera_registry, 'calibration_source_id', lambda _: 'source')
    monkeypatch.setattr(refresh, 'get_road_region', lambda _: None)
    monkeypatch.setattr(refresh.calibration_store, 'load_observations', lambda _: [])
    calls = []
    def infer(frames, width, target, height, **kw):
        calls.append((frames, kw))
        assert kw['allow_estimated_wall_width'] is True
        assert kw['cctv_id'] == 'camera'
        return ReadingCore(width, 1.8, width - 1.8, verdict={'pump-3.5': 'PASS'})
    monkeypatch.setattr(refresh, 'run_motion_aware_demo' if video else 'process_frame', infer)
    reading = {'cctv_id': 'camera', 'edge_id': 'edge', 'still_public_url': str(original),
               'source_meta': {'adapter': 'video' if video else 'file'}}
    local = refresh.refresh_readings([reading])[0]
    remote = refresh.refresh_readings([reading], use_s3=True)[0]
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    if video:
        assert len(calls[1][0]) == 10
    assert local['verdict'] == remote['verdict']
    assert local['effective_width_m'] == remote['effective_width_m']
    assert remote['still_public_url'] is None
    assert remote['source_meta']['s3_media'] == media
    assert remote['source_meta']['calibration_source_cctv_id'] == 'source'
    assert 'X-Amz' not in str(remote)
    assert refresh.refresh_readings([remote], use_s3=True)[0]['verdict'] == remote['verdict']
    def unavailable(*a, **kw):
        raise MeasurementUnavailableError('no_road_detections', 'no evidence')
    monkeypatch.setattr(refresh, 'run_motion_aware_demo' if video else 'process_frame', unavailable)
    failed = refresh.refresh_readings([remote], use_s3=True)[0]
    assert failed['verdict'] == {'pump-3.5': 'UNCERTAIN'}
    assert failed['measured_at'] == remote['measured_at']
    assert failed['source_meta']['measurement_failure']['input_transport'] == 's3_presigned'
