"""기존 S3 객체를 읽기 검증하여 모란 AI 인계 데이터에 연결한다. 업로드하지 않는다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_moran_bundle import EXPECTED_IDS, export_bundle
from src.adapters.s3_adapter import downloaded_media, validate_media
from src.output.atomic_writer import write_text_atomic

ROOT = Path(__file__).resolve().parents[1]


def link_media(readings, keys, inventory, *, bucket, region, profile=None):
    export_bundle(readings)
    files = inventory['files']
    indexed = {item['cctv_id']: item for item in files}
    if set(keys) != EXPECTED_IDS or set(indexed) != EXPECTED_IDS or len(files) != 12:
        raise ValueError('키 목록·원본 목록에 모란 CCTV 12개가 정확히 필요합니다')
    if len({item['key'] for item in keys.values()}) != 12:
        raise ValueError('서로 다른 CCTV가 같은 S3 객체를 가리킵니다')
    updated = deepcopy(readings)
    jobs = []
    for reading in updated:
        cid = reading['cctv_id']
        entry, original = keys[cid], indexed[cid]
        key = entry['key']
        if not isinstance(key, str) or not key.startswith('uploads/') or '?' in key or '#' in key:
            raise ValueError('기존 업로드의 서명 없는 객체 key가 필요합니다')
        if entry['content_type'] != original['content_type']:
            raise ValueError(f'{cid}: 업로드 목록과 원본 유형 불일치')
        meta = reading['source_meta']
        source = meta.get('original_source_url') or reading['still_public_url']
        media = {field: original[field] for field in ('original_filename', 'sha256', 'size_bytes', 'content_type')}
        media.update(bucket=bucket, region=region, key=key)
        validate_media(media, source, meta['adapter'])
        meta['original_source_url'] = source
        meta['s3_media'] = media
        reading['still_public_url'] = None
        jobs.append((media, source, meta['adapter']))
    # 모든 등록 정보가 유효한지 먼저 확인한 뒤 GET만 수행한다.
    for media, source, adapter in jobs:
        with downloaded_media(media, source, adapter, profile_name=profile):
            pass
    export_bundle(updated)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readings', type=Path, default=ROOT / 'data/cctv_readings_moran.json')
    parser.add_argument('--keys', type=Path, default=ROOT / 'data/moran_s3_upload_keys.json')
    parser.add_argument('--inventory', type=Path, default=ROOT / 'data/moran_media_inventory.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--region', default='ap-northeast-2')
    parser.add_argument('--profile')
    args = parser.parse_args()
    result = link_media(*(json.loads(p.read_text()) for p in (args.readings, args.keys, args.inventory)),
                        bucket=args.bucket, region=args.region, profile=args.profile)
    write_text_atomic(args.out, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print('S3 GET·SHA256 검증 완료: 12개. 판정·측정 시각·좌표 보존.')


if __name__ == '__main__':
    main()
