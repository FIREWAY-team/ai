import json
from pathlib import Path

import pytest

from scripts.package_moran_runtime import package_runtime, sha256, verify_package


@pytest.mark.parametrize('failure', ['missing', 'changed', 'escape'])
def test_verify_rejects_missing_changed_or_external_files(tmp_path, failure):
    root = tmp_path / 'runtime'
    root.mkdir()
    source = root / 'model.pt'
    source.write_bytes(b'trained model')
    files = {'model.pt': sha256(source)}
    if failure == 'missing': source.unlink()
    elif failure == 'changed': source.write_bytes(b'other model')
    else:
        external = tmp_path / 'external'
        external.write_bytes(b'external')
        files = {'../external': sha256(external)}
    (root / 'runtime_manifest.json').write_text(json.dumps({'files': files}))
    with pytest.raises(ValueError, match='누락/변경'):
        verify_package(root)


def test_valid_files_verify(tmp_path):
    source = tmp_path / 'model.pt'
    source.write_bytes(b'model')
    (tmp_path / 'runtime_manifest.json').write_text(json.dumps({'files': {'model.pt': sha256(source)}}))
    verify_package(tmp_path)


def test_existing_directory_is_not_overwritten(tmp_path):
    sentinel = tmp_path / 'existing'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        package_runtime(tmp_path)
    assert sentinel.read_text() == 'keep'
