"""현재 12개 판정 재현에 필요한 코드·원본·활성 보정치를 로컬 폴더로 묶는다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src import camera_registry
from src.inference import calibration_store
from scripts.export_moran_bundle import export_bundle

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(root: Path) -> None:
    root = root.resolve()
    manifest = json.loads((root / 'runtime_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'실행 묶음 파일 누락/변경: {name}')


def package_runtime(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError('기존 폴더를 덮어쓰지 않습니다')
    readings = json.loads((ROOT / 'data/cctv_readings_moran.json').read_text())
    export_bundle(readings)  # 12개 ID·좌표·판정 상태 검사
    cameras = camera_registry.load_cameras()
    selected = {r['cctv_id'] for r in readings}
    selected.update(camera_registry.calibration_source_id(r['cctv_id']) for r in readings)
    config = {key: deepcopy(cameras[key]) for key in sorted(selected)}
    histories = set()
    observation_count = 0
    for reading in readings:
        path = calibration_store._path_for(reading['cctv_id'], calibration_store.DEFAULT_CALIBRATION_DIR)
        observations = calibration_store.load_observations(reading['cctv_id'])
        expected = reading['source_meta'].get('calibration_observations_available')
        if expected != len(observations):
            raise ValueError(f"{reading['cctv_id']}: 판정 당시 보정 관측치 수 불일치")
        if path.exists() and path not in histories:
            histories.add(path)
            observation_count += len(observations)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.fireway-runtime-', dir=output.parent) as staging:
        target = Path(staging) / 'runtime'
        target.mkdir()
        def copy(source: Path, relative: Path):
            if not source.is_file():
                raise FileNotFoundError(f'필수 파일 누락: {source}')
            destination = target / relative
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError(f'파일명 충돌: {relative}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for name in ('src', 'scripts'):
            for path in (ROOT / name).rglob('*'):
                if path.suffix in {'.py', '.html'} and '__pycache__' not in path.parts:
                    copy(path, path.relative_to(ROOT))
        for name in ('yolo11n-seg.pt', 'goldenlane_vehicle_specs.py', 'requirements.txt',
                     'configs/road_regions.yaml', 'configs/vehicles.json', 'configs/moran_destination.json', 'configs/moran_demo_scenario.json',
                     'requirements-lock-macos-arm64-py313.txt'):
            copy(ROOT / name, Path(name))
        for camera in config.values():
            source = Path(camera['still_url'])
            if not source.is_absolute(): source = ROOT / source
            relative = Path('data/raw_images') / source.name
            copy(source, relative)
            camera['still_url'] = relative.as_posix()
        for path in histories:
            copy(path, path.relative_to(ROOT))
        for reading in readings:
            path = config[reading['cctv_id']]['still_url']
            reading['still_public_url'] = path
            reading['source_meta']['original_source_url'] = path
        (target / 'configs/cameras.yaml').write_text(yaml.safe_dump({'cameras': config}, allow_unicode=True, sort_keys=False))
        (target / 'data/cctv_readings_moran.json').write_text(json.dumps(readings, ensure_ascii=False, indent=2))
        versions = {}
        for name in ('numpy', 'opencv-python', 'opencv-python-headless', 'ultralytics', 'torch',
                     'torchvision', 'PyYAML', 'requests', 'boto3'):
            try: versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError: pass
        manifest = {'format': 'fireway-runtime-v1', 'camera_count': 12,
                    'calibration_file_count': len(histories), 'observation_count': observation_count,
                    'scale_version': calibration_store.SCALE_VERSION,
                    'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions,
                    'files': {p.relative_to(target).as_posix(): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}}
        (target / 'runtime_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        verify_package(target)
        target.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_package(args.verify)
        print('실행 묶음 해시 검증 완료')
    else:
        result = package_runtime(args.out)
        print(f"카메라 {result['camera_count']} / 활성 보정 파일 {result['calibration_file_count']} / 관측치 {result['observation_count']}")
        print(args.out)


if __name__ == '__main__':
    main()
