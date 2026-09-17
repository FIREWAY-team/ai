"""CCTV와 진입곤란 선형 데이터의 공간 근접 후보. 도로 매핑을 확정하지 않는다."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def distance_to_line_m(lon, lat, coordinates):
    scale = math.cos(math.radians(lat))
    points = [((x-lon)*111195*scale, (y-lat)*111195) for x,y in coordinates]
    distances = []
    for (a,b),(c,d) in zip(points, points[1:]):
        x,y=c-a,d-b
        t=max(0,min(1,-(a*x+b*y)/(x*x+y*y))) if x*x+y*y else 0
        distances.append(math.hypot(a+t*x,b+t*y))
    return min(distances, default=math.inf)


def review(bundle, roads):
    rows=[]; groups=defaultdict(list)
    lines=[r for r in roads['features'] if r['geometry']['type']=='LineString']
    for camera in bundle['features']:
        lon,lat=camera['geometry']['coordinates']; candidates=[]
        for road in lines:
            distance=distance_to_line_m(lon,lat,road['geometry']['coordinates'])
            if not math.isfinite(distance):continue
            p=road['properties']
            candidates.append({'candidate_ext_id':p['id'],'distance_m':distance,
                               'source_p90_error_m':p.get('georef_p90_error_m'),
                               'source_note':p.get('note'), 'geometry':road['geometry']})
        candidates.sort(key=lambda c:(c['distance_m'],c['candidate_ext_id']))
        top=candidates[:3]; verdict=camera['properties']['measurement']['verdict']
        row={'cctv_id':camera['id'],'verdict':verdict,'edge_id':None,'mapping_status':'unconfirmed',
             'nearest_within_review_distance':bool(top and top[0]['distance_m']<=30),
             'candidates':top}
        if top:
            groups[top[0]['candidate_ext_id']].append({'cctv_id':camera['id'],'verdict':verdict})
        rows.append(row)
    overlaps=[]
    for edge,cameras in groups.items():
        if len(cameras)<2:continue
        vehicles={v for c in cameras for v in c['verdict']}
        mixed=any(len({c['verdict'].get(v) for c in cameras})>1 for v in vehicles)
        overlaps.append({'candidate_ext_id':edge,'cameras':cameras,'mixed_verdicts':mixed})
    return {'routing_ready':False,'confirmed_mapping_count':0,'review_distance_m':30,
            'review_distance_note':'검토 편의를 위한 거리 표시 기준. 매핑 확정이나 안전 기준이 아님.',
            'source_scope':'진입곤란 지정 구간 일부. 전체 도로망이나 라우팅 엔진 edge 목록이 아님.',
            'distance_method':'CCTV 중심 국소 평면 근사에서 점-선분 최단거리; 좌표 단위 WGS84',
            'camera_candidates':rows,'nearest_candidate_overlaps':overlaps}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--roads',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.resolve() in {args.bundle.resolve(),args.roads.resolve()}:
        parser.error('출력은 입력과 다른 파일이어야 합니다')
    raw=args.roads.read_bytes()
    result=review(json.loads(args.bundle.read_text()),json.loads(raw))
    result['source_sha256']=hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print('후보 검토',len(result['camera_candidates']),'개 / 확정 매핑 0개')


if __name__=='__main__':main()
