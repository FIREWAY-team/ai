from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.export_moran_bundle import export_bundle


@pytest.fixture
def readings():
    path = Path(__file__).resolve().parents[1] / 'data/cctv_readings_moran.json'
    return json.loads(path.read_text())


def test_real_camera_points_preserve_verdicts_without_fake_road_mapping(readings):
    before = deepcopy(readings)
    bundle = export_bundle(readings)
    assert len(bundle['features']) == 12
    assert bundle['features'][0]['geometry']['coordinates'] == [127.126909, 37.430393]
    assert not bundle['routing_ready']
    assert bundle['destination']['name'] == '천안기름집'
    assert bundle['destination']['lat'] == 37.4309076894368
    assert bundle['destination']['lon'] == 127.128007683821
    assert bundle['summary']['nearby_camera_count'] == 12
    for reading, feature in zip(readings, bundle['features']):
        prop = feature['properties']
        assert prop['measurement']['verdict'] == reading['verdict']
        assert prop['edge_id'] is None
        assert prop['edge_mapping_status'] == 'unmapped'
        if prop['measurement_status'] == 'unavailable':
            assert prop['numeric_values_status'] == 'last_known'
            assert prop['measurement']['measured_at'] == reading['measured_at']
            assert prop['measurement']['effective_width_m'] == reading['effective_width_m']
    assert '/Users/' not in json.dumps(bundle)
    assert readings == before


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'foreign', 'nan', 'bad_lat', 'unavailable_pass', 'blocked_pass'])
def test_rejects_incomplete_or_conflicting_bundle(readings, change):
    if change == 'duplicate':
        readings[1] = deepcopy(readings[0])
    elif change == 'missing':
        readings.pop()
    elif change == 'foreign':
        readings[0]['cctv_id'] = 'other'
    elif change == 'nan':
        readings[0]['source_meta']['lon'] = float('nan')
    elif change == 'bad_lat':
        readings[0]['source_meta']['lat'] = 127
    else:
        readings[0]['verdict'] = {'pump-3.5': 'PASS'}
        readings[0]['source_meta']['measurement_status'] = 'unavailable' if change == 'unavailable_pass' else 'computed'
        readings[0]['source_meta']['measurement_quality'] = {'pass_blocked': change == 'blocked_pass'}
    with pytest.raises(ValueError):
        export_bundle(readings)


def test_private_media_reference_without_temporary_url(readings):
    first = readings[0]
    original = first['still_public_url']
    first['source_meta']['original_source_url'] = original
    first['still_public_url'] = None
    first['source_meta']['s3_media'] = {
        'bucket': 'private', 'key': 'cctv/uuid.png', 'region': 'ap-northeast-2',
        'content_type': 'image/png', 'size_bytes': 100, 'sha256': 'a' * 64,
        'original_filename': Path(original).name,
        'temporary_url': 'https://example.test/?X-Amz-Signature=secret',
    }
    bundle = export_bundle(readings)
    prop = bundle['features'][0]['properties']
    assert prop['media_status'] == 'registered'
    assert prop['media']['key'] == 'cctv/uuid.png'
    assert 'X-Amz' not in json.dumps(bundle)
    assert bundle['summary']['registered_media_count'] == 1


def test_radius_selection_keeps_outside_camera_without_moving_it(readings):
    readings[0]['source_meta']['lat'] = 37.44
    bundle = export_bundle(readings)
    first = bundle['features'][0]
    assert first['geometry']['coordinates'] == [127.126909, 37.44]
    assert not first['properties']['within_destination_radius']
    assert first['properties']['distance_to_destination_m'] > 200
    assert len(bundle['features']) == 12
    assert bundle['summary']['nearby_camera_count'] == 11


@pytest.mark.parametrize('field,value', [('lat', float('nan')), ('lon', 181), ('radius_m', 0)])
def test_invalid_destination_is_rejected(readings, tmp_path, monkeypatch, field, value):
    import scripts.export_moran_bundle as exporter
    destination = json.loads(exporter.DESTINATION_PATH.read_text())
    destination[field] = value
    path = tmp_path / 'destination.json'
    path.write_text(json.dumps(destination))
    monkeypatch.setattr(exporter, 'DESTINATION_PATH', path)
    with pytest.raises(ValueError):
        exporter.export_bundle(readings)
