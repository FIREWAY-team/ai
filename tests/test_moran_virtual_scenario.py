from copy import deepcopy
import json

import pytest

from scripts.build_moran_virtual_scenario import ROOT, SCENARIO_PATH, build_virtual_scenario


def test_virtual_scenario_preserves_evidence_and_distinguishes_vehicle_widths():
    rows = json.loads((ROOT / 'data/cctv_readings_moran.json').read_text())
    before = deepcopy(rows)
    scenario = json.loads(SCENARIO_PATH.read_text())
    bundle = build_virtual_scenario(rows, scenario)
    assert rows == before
    assert bundle['demo_only'] and not bundle['routing_ready']
    counts = bundle['summary']['verdict_counts']
    assert counts['pump-3.5'] == {'PASS': 6, 'FAIL': 3, 'UNCERTAIN': 3}
    assert counts['pump-8'] == counts['aerial-25'] == {'PASS': 5, 'FAIL': 3, 'UNCERTAIN': 4}
    assert counts['pump-15'] == {'PASS': 4, 'FAIL': 6, 'UNCERTAIN': 2}
    by_id = {r['cctv_id']: r for r in rows}
    for feature in bundle['features']:
        prop = feature['properties']; original = by_id[feature['id']]
        assert prop['baseline_evidence']['measurement']['verdict'] == original['verdict']
        assert feature['geometry']['coordinates'] == [original['source_meta']['lon'], original['source_meta']['lat']]
        assert prop['edge_id'] is None
        assert prop['measurement']['confidence'] is None
        assert prop['simulation']['media_is_measurement_evidence'] is False
        if feature['id'] == 'cctv_moran_a54':
            assert set(prop['measurement']['verdict'].values()) == {'UNCERTAIN'}
            assert prop['measurement_status'] == 'unavailable'


@pytest.mark.parametrize('bad', [True, -1, 0, float('nan'), float('inf'), '3.2'])
def test_invalid_virtual_width_rejected(bad):
    scenario = json.loads(SCENARIO_PATH.read_text())
    scenario['effective_widths_m']['cctv_moran_a41'] = bad
    with pytest.raises(ValueError):
        build_virtual_scenario([], scenario)
