"""S3만 메모리로 대체한 업로드→설정→디코딩→판정→GeoJSON 연결 검증."""
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
import requests
import yaml

from scripts import export_moran_bundle as exporter, refresh_moran_readings as refresh, register_camera
from src import camera_registry
from src.adapters import s3_adapter
from src.output import atomic_writer, s3_uploader
from src.schemas import ReadingCore


class MemoryS3:
    meta = type('Meta', (), {'region_name': 'ap-northeast-2'})()

    def __init__(self):
        self.objects = {}
        self.signatures = 0
        self.downloads = 0
        self.fail_at = None
        self.expire_once = False
        self.corrupt_at = None

    def upload_file(self, path, bucket, key, ExtraArgs):
        self.objects[key] = Path(path).read_bytes()

    def generate_presigned_url(self, operation, Params, ExpiresIn, HttpMethod):
        self.signatures += 1
        return f"https://offline.test/{Params['Key']}?X-Amz-Signature=secret-{self.signatures}"

    def get(self, url, **kwargs):
        self.downloads += 1
        if self.downloads == self.fail_at:
            raise requests.Timeout('secret URL must not leak')
        data = self.objects[urlsplit(url).path.lstrip('/')]
        status = 403 if self.expire_once and self.downloads == 1 else 200
        if self.downloads == self.corrupt_at:
            data = bytes(len(data))
        class Response:
            status_code = status
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, chunk_size): yield data
        return Response()


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    original = json.loads((Path(__file__).resolve().parents[1] / 'data/cctv_readings_moran.json').read_text())
    cameras = {}
    frames = np.full((32, 48, 3), 100, np.uint8)
    video = tmp_path / 'sample.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 5, (48, 32))
    assert writer.isOpened()
    for i in range(50): writer.write(frames)
    writer.release()
    for i, reading in enumerate(original):
        path = tmp_path / (reading['cctv_id'] + ('.png' if i < 2 else '.mp4'))
        if i < 2: cv2.imwrite(str(path), frames)
        else: path.write_bytes(video.read_bytes())
        camera = {'still_url': str(path), 'wall_width_m': 6.08,
                  'wall_width_source': 'estimated_avg_of_similar_alleys'}
        cameras[reading['cctv_id']] = camera
        reading['still_public_url'] = str(path)
        reading['source_meta'].pop('original_source_url', None)
        reading['source_meta'].pop('s3_media', None)
        reading['source_meta']['adapter'] = 'file' if i < 2 else 'video'
    first_id = original[0]['cctv_id']
    cameras['source'] = deepcopy(cameras[first_id])
    cameras[first_id]['calibration_source_cctv_id'] = 'source'
    registry = tmp_path / 'cameras.yaml'
    registry.write_text(yaml.safe_dump({'cameras': cameras}))
    real_load = camera_registry.load_cameras
    monkeypatch.setattr(camera_registry, 'load_cameras', lambda path=None: real_load(registry))
    monkeypatch.setattr(refresh.calibration_store, 'load_observations', lambda _: [])
    monkeypatch.setattr(refresh, 'get_road_region', lambda _: None)
    calls = []
    def infer(frames, width, target, height, **kw):
        assert kw['allow_estimated_wall_width'] is True
        if isinstance(frames, list): assert len(frames) == 10
        calls.append(kw['cctv_id'])
        return ReadingCore(width, 1.8, width - 1.8, verdict={'pump-3.5': 'PASS', 'pump-8': 'PASS'})
    monkeypatch.setattr(refresh, 'process_frame', infer)
    monkeypatch.setattr(refresh, 'run_motion_aware_demo', infer)
    s3 = MemoryS3()
    monkeypatch.setattr(s3_uploader, '_get_client', lambda *a: s3)
    monkeypatch.setattr(s3_adapter.requests, 'get', s3.get)
    seed = tmp_path / 'seed.json'
    seed.write_text(json.dumps(original))
    return original, registry, seed, s3, calls


def register_all(workflow):
    original, registry, _, _, _ = workflow
    for reading in original:
        register_camera.register_one(
            register_camera.CameraEntry(reading['cctv_id'], reading['still_public_url'], 6.08),
            'private', 'ap-northeast-2', 'kakao_map', 'low', registry_path=registry,
        )


def run_refresh(monkeypatch, seed, output):
    monkeypatch.setattr('sys.argv', ['refresh', '--source', 's3', '--readings', str(seed), '--out', str(output)])
    refresh.main()


def test_twelve_media_upload_refresh_export_and_expired_url(workflow, monkeypatch, tmp_path):
    original, registry, seed, s3, calls = workflow
    register_all(workflow)
    assert len(s3.objects) == 12
    cameras = yaml.safe_load(registry.read_text())['cameras']
    assert cameras[original[0]['cctv_id']]['calibration_source_cctv_id'] == 'source'
    assert 's3_media' not in cameras['source']
    assert cameras[original[0]['cctv_id']]['wall_width_source'] == 'estimated_avg_of_similar_alleys'
    s3.expire_once = True
    output = tmp_path / 'result.json'
    run_refresh(monkeypatch, seed, output)
    assert len(calls) == 12
    assert s3.signatures == 13
    bundle_path = tmp_path / 'bundle.geojson'
    monkeypatch.setattr('sys.argv', ['export', '--readings', str(output), '--out', str(bundle_path)])
    exporter.main()
    bundle = json.loads(bundle_path.read_text())
    assert bundle['summary']['registered_media_count'] == 12
    assert not bundle['routing_ready']
    types = [f['properties']['media_type'] for f in bundle['features']]
    assert types.count('image') == 2 and types.count('video') == 10
    for path in [registry, output, bundle_path]:
        assert 'X-Amz-Signature' not in path.read_text()
    # 생성한 S3 결과를 다시 입력해도 원본 식별/보정 연결이 유지된다.
    run_refresh(monkeypatch, output, output)
    assert len(calls) == 24


@pytest.mark.parametrize('failure', ['timeout', 'hash'])
def test_mid_batch_download_failure_preserves_existing_outputs(workflow, monkeypatch, tmp_path, failure):
    _, registry, seed, s3, calls = workflow
    register_all(workflow)
    registry_before = registry.read_bytes()
    output = tmp_path / 'result.json'
    output.write_bytes(b'previous reading file')
    bundle = tmp_path / 'bundle.geojson'
    bundle.write_bytes(b'previous bundle')
    if failure == 'timeout': s3.fail_at = 7
    else: s3.corrupt_at = 7
    with pytest.raises((s3_adapter.MediaDownloadError, ValueError)):
        run_refresh(monkeypatch, seed, output)
    assert len(calls) == 6
    assert output.read_bytes() == b'previous reading file'
    assert bundle.read_bytes() == b'previous bundle'
    assert registry.read_bytes() == registry_before


@pytest.mark.parametrize('operation', ['register', 'refresh', 'export'])
def test_disk_replace_failure_preserves_existing_file(workflow, monkeypatch, tmp_path, operation):
    original, registry, seed, s3, _ = workflow
    register_all(workflow)
    target = registry if operation == 'register' else tmp_path / 'existing.json'
    if operation != 'register': target.write_bytes(b'previous output')
    before = target.read_bytes()
    def fail(*args): raise OSError('disk replacement denied')
    monkeypatch.setattr(atomic_writer.os, 'replace', fail)
    with pytest.raises(OSError):
        if operation == 'register':
            reading = original[0]
            register_camera.register_one(register_camera.CameraEntry(reading['cctv_id'], reading['still_public_url'], 6.08),
                                         'private', 'ap-northeast-2', 'kakao_map', 'low', registry_path=registry)
        elif operation == 'refresh':
            run_refresh(monkeypatch, seed, target)
        else:
            monkeypatch.setattr('sys.argv', ['export', '--readings', str(seed), '--out', str(target)])
            exporter.main()
    assert target.read_bytes() == before
    assert not list(target.parent.glob(f'.{target.name}.*'))


def test_registration_cli_reports_failure_exit_status(monkeypatch, tmp_path):
    def fail(*a, **kw): raise OSError('upload failed')
    monkeypatch.setattr(register_camera, 'register_one', fail)
    monkeypatch.setattr('sys.argv', ['register', '--bucket', 'private', '--cctv-id', 'cam',
                                    '--image', str(tmp_path / 'image.png'), '--wall-width-m', '5'])
    with pytest.raises(SystemExit) as error:
        register_camera.main()
    assert error.value.code == 1
