from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest
from scripts import link_moran_media as linker


def inputs():
    readings = json.loads((Path(__file__).resolve().parents[1] / 'data/cctv_readings_moran.json').read_text())
    keys, files = {}, []
    for n, r in enumerate(readings):
        source = r['source_meta'].get('original_source_url') or r['still_public_url']
        content_type = 'video/mp4' if r['source_meta']['adapter'] == 'video' else 'image/png'
        keys[r['cctv_id']] = {'key': f'uploads/test/{n}', 'content_type': content_type}
        files.append(dict(cctv_id=r['cctv_id'], original_filename=Path(source).name,
                          content_type=content_type, size_bytes=1, sha256='a'*64))
    return readings, keys, {'files': files}


def test_link_preserves_measurements_and_input(monkeypatch):
    args = inputs(); before = deepcopy(args); calls = []
    @contextmanager
    def download(*a, **kw):
        calls.append(a); yield 'verified'
    monkeypatch.setattr(linker, 'downloaded_media', download)
    result = linker.link_media(*args, bucket='private', region='region')
    assert args == before
    assert len(calls) == 12
    for old, new in zip(args[0], result):
        assert new['still_public_url'] is None
        for key in old.keys() - {'still_public_url', 'source_meta'}:
            assert new[key] == old[key]
        for key in old['source_meta'].keys() - {'s3_media', 'original_source_url'}:
            assert new['source_meta'][key] == old['source_meta'][key]
    assert linker.export_bundle(result)['summary']['registered_media_count'] == 12


def test_bad_mapping_rejected_before_network(monkeypatch):
    args = inputs(); args[1].pop(next(iter(args[1])))
    monkeypatch.setattr(linker, 'downloaded_media', lambda *a, **k: pytest.fail('no GET expected'))
    with pytest.raises(ValueError):
        linker.link_media(*args, bucket='private', region='region')


def test_failed_download_preserves_input(monkeypatch):
    args = inputs(); before = deepcopy(args)
    @contextmanager
    def fail(*a, **kw):
        raise RuntimeError('GET failed')
        yield
    monkeypatch.setattr(linker, 'downloaded_media', fail)
    with pytest.raises(RuntimeError):
        linker.link_media(*args, bucket='private', region='region')
    assert args == before
