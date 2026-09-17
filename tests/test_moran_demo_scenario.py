from copy import deepcopy
import json
from pathlib import Path

import pytest
from scripts.build_moran_demo_scenario import build_scenario, SCENARIO_PATH
from scripts.export_moran_bundle import export_bundle


@pytest.fixture
def case():
    rows = json.loads((Path(__file__).resolve().parents[1] / 'data/cctv_readings_moran.json').read_text())
    for r in rows:
        m = r['source_meta']; source = m.get('original_source_url') or r['still_public_url']
        m['s3_media'] = dict(bucket='test', key=r['cctv_id'], region='region',
                            original_filename=Path(source).name, size_bytes=1, sha256='a'*64,
                            content_type='video/mp4' if m['adapter']=='video' else 'image/png')
    return rows, json.loads(SCENARIO_PATH.read_text())


def test_demo_preserves_locations_baseline_and_copies_full_evidence(case):
    rows, config = case; before = deepcopy(rows)
    baseline = export_bundle(rows); result = build_scenario(rows, config)
    assert rows == before
    assert result['demo_only'] and not result['routing_ready']
    assert result['summary']['unique_media_count'] == 8
    assert result['baseline_summary']['verdict_counts']['pump-8']['PASS'] == 1
    assert result['summary']['verdict_counts']['pump-8'] == {'PASS':5,'FAIL':3,'UNCERTAIN':4}
    donor = next(f for f in baseline['features'] if f['id']==config['source_cctv_id'])['properties']
    for original, changed in zip(baseline['features'],result['features']):
        p = changed['properties']
        assert original['geometry'] == changed['geometry']
        assert p['baseline_evidence']['measurement'] == original['properties']['measurement']
        assert p['edge_id'] is None
        if changed['id'] in config['reuse_at_cctv_ids']:
            for key in ('measurement','media','provenance'):
                assert p[key] == donor[key]
        else:
            assert p['measurement'] == original['properties']['measurement']
    assert 'X-Amz-' not in json.dumps(result)


@pytest.mark.parametrize('bad', ['source_fail','target_fail','target_unavailable','duplicate','source_missing_media'])
def test_reject_invalid_evidence_or_placement(case,bad):
    rows,config=case
    donor=next(r for r in rows if r['cctv_id']==config['source_cctv_id'])
    if bad=='source_fail':donor['verdict']['pump-8']='FAIL'
    elif bad=='source_missing_media':donor['source_meta'].pop('s3_media')
    elif bad=='target_fail':config['reuse_at_cctv_ids'][0]='cctv_moran_a18'
    elif bad=='target_unavailable':config['reuse_at_cctv_ids'][0]='cctv_moran_a54'
    else:config['reuse_at_cctv_ids'][0]=config['reuse_at_cctv_ids'][1]
    with pytest.raises(ValueError):build_scenario(rows,config)
