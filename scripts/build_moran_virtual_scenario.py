"""실제 영상 판독을 보존한 별도 가상 폭 시나리오. 운영 DB는 변경하지 않는다."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldenlane_vehicle_specs import load_vehicles_json, resolve_margin_m
from scripts.export_moran_bundle import EXPECTED_IDS, export_bundle
from src.output.atomic_writer import write_text_atomic
from src.postprocess.passable_prob import build_reading_verdict

SCENARIO_PATH = ROOT / 'configs/moran_virtual_width_scenario.json'


def build_virtual_scenario(readings, scenario):
    widths = scenario['effective_widths_m']
    if set(widths) != EXPECTED_IDS or not scenario.get('scenario_id'):
        raise ValueError('시나리오 ID 및 CCTV 12개 가상 폭이 필요합니다')
    if any(w is not None and (type(w) not in (int, float) or not math.isfinite(w) or w <= 0)
           for w in widths.values()):
        raise ValueError('가상 폭은 양의 유한 숫자 또는 null이어야 합니다')
    bundle = export_bundle(readings)
    vehicles = load_vehicles_json()
    margin = resolve_margin_m()
    bundle['baseline_summary'] = deepcopy(bundle['summary'])
    for feature in bundle['features']:
        prop = feature['properties']
        prop['baseline_evidence'] = deepcopy(prop)
        width = widths[feature['id']]
        verdicts = ({vehicle: 'UNCERTAIN' for vehicle in vehicles} if width is None
                    else build_reading_verdict(width, vehicles, margin, 0.0)[0])
        # 영상은 배경 자료일 뿐 가상 폭·판정의 측정 근거가 아니다.
        prop['measurement'] = dict(wall_width_m=None, obstacle_width_m=None,
                                   effective_width_m=width, calibration_error_m=0.0 if width else None,
                                   confidence=None, measured_at=None, method='virtual_width_simulation',
                                   verdict=verdicts)
        prop['measurement_status'] = 'unavailable' if width is None else 'simulated'
        prop['numeric_values_status'] = 'unavailable' if width is None else 'simulated'
        prop['provenance'] = {'wall_width_source': 'synthetic', 'measurement_scope': 'virtual_scenario'}
        prop['simulation'] = {'scenario_id': scenario['scenario_id'], 'synthetic': True,
                              'validates_target_road': False, 'media_is_measurement_evidence': False,
                              'margin_m': margin, 'calibration_error_m': 0.0}
        prop['footage_note'] = '영상은 참고용. 폭과 판정은 별도 가상 시나리오이며 실제 골목 측정값이 아닙니다.'
    bundle.update(demo_only=True, scenario=deepcopy(scenario), vehicles=vehicles,
                  schema_version='fireway-virtual-width-v1', routing_ready=False)
    bundle['summary']['verdict_counts'] = {
        v: dict(Counter(f['properties']['measurement']['verdict'][v] for f in bundle['features']))
        for v in vehicles}
    bundle['limitations'].extend(['가상 유효폭으로만 판정. 실제 통행 가능성 검증 아님',
                                  '동일 폭의 중형펌프차와 굴절차는 동일 판정',
                                  '도로 매핑 및 백엔드 시나리오 연결 전 경로 안내에 미반영'])
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readings', type=Path, default=ROOT / 'data/cctv_readings_moran.json')
    parser.add_argument('--scenario', type=Path, default=SCENARIO_PATH)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/moran_virtual_width_bundle.geojson')
    args = parser.parse_args()
    if args.out.resolve() in {args.readings.resolve(), args.scenario.resolve()} or not args.out.name.endswith('_virtual_width_bundle.geojson'):
        parser.error('별도 *_virtual_width_bundle.geojson 출력만 허용합니다')
    result = build_virtual_scenario(json.loads(args.readings.read_text()), json.loads(args.scenario.read_text()))
    write_text_atomic(args.out, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
