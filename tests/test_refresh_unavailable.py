from copy import deepcopy

import numpy as np
import pytest

from scripts import refresh_moran_readings as refresh
from src.schemas import MeasurementUnavailableError, ReadingCore


@pytest.mark.parametrize('adapter', ['file', 'video'])
def test_unavailable_preserves_previous_measurement_and_recovers(monkeypatch, adapter):
    original = {'cctv_id': 'camera', 'edge_id': 'edge', 'still_public_url': 'scene',
                'measured_at': 'previous-time', 'effective_width_m': 4.28,
                'verdict': {'pump': 'PASS'}, 'confidence': .9,
                'source_meta': {'adapter': adapter}}
    before = deepcopy(original)
    monkeypatch.setattr(refresh.camera_registry, 'get_camera', lambda _: {'still_url': 'scene', 'wall_width_m': 6.08})
    monkeypatch.setattr(refresh.camera_registry, 'calibration_source_id', lambda _: 'camera')
    monkeypatch.setattr(refresh, 'get_road_region', lambda _: None)
    monkeypatch.setattr(refresh.calibration_store, 'load_observations', lambda _: [])
    monkeypatch.setattr(refresh, 'load_frame', lambda _: np.zeros((20, 20, 3)))
    monkeypatch.setattr(refresh, 'extract_frames', lambda *a, **kw: [np.zeros((20, 20, 3))] * 10)
    def unavailable(*a, **kw):
        raise MeasurementUnavailableError('no_road_detections', 'no evidence')
    processor = 'process_frame' if adapter == 'file' else 'run_motion_aware_demo'
    monkeypatch.setattr(refresh, processor, unavailable)
    result = refresh.refresh_readings([original, original])
    assert len(result) == 2
    failed = result[0]
    assert failed['verdict'] == {'pump': 'UNCERTAIN'}
    assert failed['confidence'] == 0
    assert failed['measured_at'] == 'previous-time'
    assert failed['effective_width_m'] == 4.28
    assert failed['source_meta']['measurement_status'] == 'unavailable'
    assert failed['source_meta']['measurement_quality']['pass_blocked']
    assert original == before
    monkeypatch.setattr(refresh, processor, lambda *a, **kw: ReadingCore(6.08, 1.8, 4.28, verdict={'pump': 'PASS'}))
    recovered = refresh.refresh_readings([failed])[0]
    assert recovered['source_meta']['measurement_status'] == 'computed'
    assert 'measurement_failure' not in recovered['source_meta']
    assert recovered['source_meta']['measurement_quality'] == {'flags': [], 'pass_blocked': False}
    assert recovered['measured_at'] != 'previous-time'
    def broken(*a, **kw):
        raise ValueError('bad configuration')
    monkeypatch.setattr(refresh, processor, broken)
    with pytest.raises(ValueError, match='bad configuration'):
        refresh.refresh_readings([original])
