"""실제 판정은 보존하고 동일 PASS 영상 재사용을 명시한 별도 시연 GeoJSON 생성."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_moran_bundle import export_bundle
from src.output.atomic_writer import write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / 'configs/moran_demo_scenario.json'


def build_scenario(readings, scenario):
    bundle = export_bundle(readings)
    features = {f['id']: f for f in bundle['features']}
    source_id = scenario['source_cctv_id']
    targets = scenario['reuse_at_cctv_ids']
    if len(targets) != 4 or len(set(targets)) != 4 or source_id in targets or not set(targets) <= features.keys():
        raise ValueError('서로 다른 추가 CCTV 4개가 필요합니다')
    donor = features[source_id]['properties']
    vehicles = {'pump-3.5', 'pump-8'}
    if (donor['measurement_status'] != 'computed' or donor['media_type'] != 'video'
            or donor['media'] is None or set(donor['measurement']['verdict']) != vehicles
            or any(v != 'PASS' for v in donor['measurement']['verdict'].values())):
        raise ValueError('원본은 두 차종 모두 PASS인 등록된 영상의 유효 측정이어야 합니다')
    evidence_fields = ('measurement', 'provenance', 'media', 'media_type', 'media_status',
                       'original_filename', 'measurement_status', 'numeric_values_status')
    for cid in targets:
        target = features[cid]['properties']
        if (set(target['measurement']['verdict']) != vehicles
                or any(v != 'UNCERTAIN' for v in target['measurement']['verdict'].values())
                or target['measurement_status'] != 'computed'):
            raise ValueError('추가 배치는 측정 가능한 UNCERTAIN 지점만 허용합니다')
    for cid, feature in features.items():
        prop = feature['properties']
        prop['baseline_evidence'] = {k: deepcopy(prop[k]) for k in evidence_fields}
        reused = cid in targets
        if reused:
            for key in evidence_fields:
                prop[key] = deepcopy(donor[key])
        prop['demo_assignment'] = {
            'scenario_id': scenario['scenario_id'], 'evidence_cctv_id': source_id if reused else cid,
            'shared_pass_footage': reused or cid == source_id,
            'reassigned': reused, 'validates_target_road': False,
            'measurement_scope': 'source_footage',
        }
        prop['footage_mode'] = 'shared_source_demo' if reused or cid == source_id else 'substitute_demo'
        prop['footage_note'] = '시연용 대체 영상 판정. 해당 설치 위치의 실제 도로 통과 가능성을 검증한 값이 아닙니다.'
    counts = {v: dict(Counter(f['properties']['measurement']['verdict'][v] for f in features.values()))
              for v in sorted(vehicles)}
    if any(c != scenario['expected_verdict_counts'] for c in counts.values()):
        raise ValueError('시연 분배와 원본 판정이 맞지 않습니다')
    bundle['baseline_summary'] = deepcopy(bundle['summary'])
    bundle['summary']['verdict_counts'] = counts
    bundle['summary']['unique_media_count'] = len({(f['properties']['media']['bucket'], f['properties']['media']['key'])
                                                  for f in features.values() if f['properties']['media']})
    bundle['scenario'] = deepcopy(scenario)
    bundle['demo_only'] = True
    bundle['limitations'].append('PASS 5곳은 동일 영상 근거 재사용이며 5개 도로의 독립 검증이 아님')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readings', type=Path, default=ROOT / 'data/moran_s3_verified_readings.json')
    parser.add_argument('--scenario', type=Path, default=SCENARIO_PATH)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/moran_demo_5pass_bundle.geojson')
    args = parser.parse_args()
    # 원본·일반 인계 출력과 혼동하지 않도록 시연 전용 파일명을 요구한다.
    if args.out.resolve() in {args.readings.resolve(), args.scenario.resolve()} or not args.out.name.endswith('_demo_5pass_bundle.geojson'):
        parser.error('출력은 별도 *_demo_5pass_bundle.geojson 파일이어야 합니다')
    result = build_scenario(json.loads(args.readings.read_text()), json.loads(args.scenario.read_text()))
    write_text_atomic(args.out, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result['summary'], ensure_ascii=False))
    print(args.out)


if __name__ == '__main__':
    main()
