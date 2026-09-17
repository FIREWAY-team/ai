"""네트워크 없이 열 수 있는 CCTV/도로 후보 위치 검토도 생성."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def render(bundle, review):
    roads={c['candidate_ext_id']:c for row in review['camera_candidates'] for c in row['candidates']}
    locations=[f['geometry']['coordinates'] for f in bundle['features']]
    locations += [xy for road in roads.values() for xy in road['geometry']['coordinates']]
    import math
    destination=bundle['destination']
    lat_delta=destination['radius_m']/111195
    lon_delta=lat_delta/math.cos(math.radians(destination['lat']))
    locations += [[destination['lon']-lon_delta,destination['lat']-lat_delta],
                  [destination['lon']+lon_delta,destination['lat']+lat_delta]]
    west,east=min(p[0] for p in locations),max(p[0] for p in locations)
    south,north=min(p[1] for p in locations),max(p[1] for p in locations)
    scale=min(850/((east-west)*math.cos(math.radians(south))),680/(north-south))
    def xy(p):return 60+(p[0]-west)*math.cos(math.radians(south))*scale,50+(north-p[1])*scale
    svg=[]
    for key,road in roads.items():
        pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in map(xy,road['geometry']['coordinates']))
        svg.append(f'<polyline data-road="{html.escape(key)}" points="{pts}" class="road"><title>{html.escape(key)}</title></polyline>')
    destination=bundle['destination'];x,y=xy([destination['lon'],destination['lat']])
    radius=destination['radius_m']/111195*scale
    svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="#ef444408" stroke="#c2410c" stroke-dasharray="5 6"/>')
    svg.append(f'<g><circle cx="{x}" cy="{y}" r="8" fill="#b91c1c"/><text x="{x+12}" y="{y-10}" class="fire">천안기름집</text></g>')
    data=[]
    by_id={r['cctv_id']:r for r in review['camera_candidates']}
    for f in bundle['features']:
        p=f['properties'];short=f['id'].removeprefix('cctv_moran_').upper();x,y=xy(f['geometry']['coordinates']);v=p['measurement']['verdict']['pump-8'];color={'PASS':'#16804a','FAIL':'#c43e39','UNCERTAIN':'#ab7300'}[v]
        svg.append(f'<g class="camera" data-camera="{f["id"]}" tabindex="0" role="button" aria-label="{short}"><circle cx="{x}" cy="{y}" r="12" fill="{color}"/><text x="{x+16}" y="{y+5}">{short}</text></g>')
        data.append(dict(id=f['id'],short=short,verdict=v,address=p['address'],source=p['demo_assignment']['evidence_cctv_id'],baseline=p['baseline_evidence']['measurement']['verdict']['pump-8'],candidates=by_id[f['id']]['candidates']))
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>천안기름집 CCTV 도로 후보 검토</title><style>
body{margin:0;background:#f1f5f9;color:#172435;font:15px system-ui}header{padding:20px 25px;background:#fff;border-bottom:1px solid #ddd}h1{font-size:23px;margin:0 0 8px}p{line-height:1.6;margin:6px 0}.layout{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:16px;padding:16px}.panel{background:white;border-radius:12px;padding:18px}svg{width:100%;max-height:76vh;background:#f8fafc}.road{fill:none;stroke:#a5acb5;stroke-width:3;stroke-linecap:round}.road.active{stroke:#2869c8;stroke-width:6}.camera{cursor:pointer}.camera circle{stroke:white;stroke-width:2}.camera.selected circle{stroke:#111827;stroke-width:4}text{font:bold 15px system-ui;paint-order:stroke;stroke:#fff;stroke-width:4px;fill:#172435}.fire{fill:#b91c1c}button{cursor:pointer;border:1px solid #ccd3dd;border-radius:6px;background:white;padding:8px;margin:3px;color:#172435}button.chosen{background:#dbeafe}li{margin:12px 0;line-height:1.5;font-size:13px}small{color:#526071}.alert{background:#fff3df;border-radius:8px;padding:10px}.legend{font-size:13px}@media(max-width:850px){.layout{grid-template-columns:1fr}svg{max-height:none}}
</style><header><h1>천안기름집 · CCTV 12곳 도로 후보 검토</h1><p>데모: PASS 5 / FAIL 3 / UNCERTAIN 4 · 확정 도로 연결 0개</p><small>실제 좌표 기반 위치 검토도 · 배경 지도 없음 · 북쪽 ↑ · 점은 시연 판정, 선은 진입곤란 지정 구간 후보</small></header><main class="layout"><section class="panel"><svg viewBox="0 0 980 820" role="img" aria-label="고정 CCTV 12개와 천안기름집 및 연결 후보 도로">'''+''.join(svg)+'''</svg><p class="legend">초록·빨강·황색 점: CCTV 데모 판정 / 회색 선: 도로 후보 / 파란 선: 선택 CCTV 최근접 3개 후보 / 주황 점선: 200m</p></section><aside class="panel"><div id="buttons"></div><h2 id="title">CCTV 선택</h2><p id="detail"></p><ol id="candidates"></ol><div class="alert">근접한 선 ≠ 확인된 담당 도로.<br>A49: 최근접 55.6m / A59: 48.5m.<br>A41·A17: 최근접 후보 중복.</div><p><small>PASS 5곳은 A41 영상 재사용. 실제 5개 도로를 독립 검증한 값이 아닙니다. 촬영 방향·담당 구간 및 최신 RDS 데이터 확인 후 매핑해야 합니다.</small></p><p><small>후보 원본: backend 75b4545 스냅샷. 최신 배포 확인 전. 클릭은 후보 비교만 하며 설정·DB에 저장하지 않습니다.</small></p></aside></main><script>
const cameras='''+payload+''';
function select(id){const p=cameras.find(x=>x.id===id);document.getElementById('title').textContent=p.short+' · '+p.verdict;document.getElementById('detail').textContent=p.address+' / 원래 판정 '+p.baseline+' / 영상 근거 '+p.source;document.querySelectorAll('[data-road]').forEach(el=>el.classList.toggle('active',p.candidates.some(c=>c.candidate_ext_id===el.dataset.road)));document.querySelectorAll('[data-camera]').forEach(el=>el.classList.toggle('selected',el.dataset.camera===id));document.querySelectorAll('#buttons button').forEach(el=>el.classList.toggle('chosen',el.dataset.id===id));const list=document.getElementById('candidates');list.replaceChildren();p.candidates.forEach(c=>{const li=document.createElement('li');li.textContent=c.candidate_ext_id+' · '+c.distance_m.toFixed(1)+'m';list.append(li);});}
for(const p of cameras){const b=document.createElement('button');b.textContent=p.short;b.dataset.id=p.id;b.onclick=()=>select(p.id);document.getElementById('buttons').append(b);}
document.querySelectorAll('[data-camera]').forEach(el=>{el.onclick=()=>select(el.dataset.camera);el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(el.dataset.camera);}};});select('cctv_moran_a49');
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=ROOT/'data/moran_demo_5pass_bundle.geojson')
    p.add_argument('--review',type=Path,default=ROOT/'data/moran_road_candidates.json')
    p.add_argument('--out',type=Path,default=ROOT/'data/location_review/moran_road_review.html')
    a=p.parse_args()
    if a.out.resolve() in {a.bundle.resolve(),a.review.resolve()}:p.error('출력은 입력과 달라야 합니다')
    output=render(json.loads(a.bundle.read_text()),json.loads(a.review.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(output,encoding='utf-8');print(a.out)

if __name__=='__main__':main()
