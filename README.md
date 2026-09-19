# ai
FireWay AI/ML

## 모란 시연: 원본 영상의 도로 폭 복원

모란 CCTV 좌표는 지도 배치용이다. 영상 판독의 `wall_width_m`은 해당 원본
영상의 도로 폭을 사용한다. `configs/cameras.yaml`의 동일 `still_url` 원본에서
`naver_map` 또는 `kakao_map` 출처인 폭을 찾아 모란 설정과 판정 JSON을 갱신한다.
원본 로컬 파일·모델 가중치·카메라 설정이 있는 환경에서 실행한다.

```sh
python scripts/restore_moran_widths.py
python scripts/refresh_moran_readings.py
python -m pytest -q
```

복원 대상: a18=5.6m, a39=7.6m, a17=28.0m, a1=6.4m, a59=6.0m.
그 외 7개는 원본 지도 측정 폭 근거가 없어 기존 추정값 유지. 현재 모란 데모는
추정 폭 사용을 허용하며 출처·적용 정책을 함께 기록한다.
현재 데모 기준 12곳은 PASS 2 / FAIL 3 / UNCERTAIN 7이다.
a17 원본은 폭 28m여서 골목 시연 적합성 검토·영상 교체 필요.

출력 `source_meta`의 `width_reference_cctv_id`는 폭 근거 원본 ID,
`wall_width_source`는 측정 출처, `wall_width_scope=source_footage`는 원본 영상
폭임을 뜻한다. 설정은 로컬 파일이며 Git 제외, 결과는
`data/cctv_readings_moran.json`에 저장한다.

폭 복원 스크립트는 현재 캘리브레이션 설정으로 재판정한다. 재판정은 기존과 같은
영상 10프레임·1초 간격을 사용하며, `measured_at`은 재판정 시각이다.

## 원본 누적 캘리브레이션 연결

`configs/cameras.yaml`의 모란 항목에 `calibration_source_cctv_id`를 지정한다.
현재 연결: a18→cctv_2, a39→cctv_3, a21→cctv_8, a17→cctv_7,
a41→cctv_9, a1→cctv_4, a59→cctv_5.

원본과 모란 항목의 `still_url`이 동일할 때만 원본 JSONL을 공유한다.
영상 경로 불일치·미등록 원본·별칭 중첩은 오류, 현재 프레임 높이와 다른
관측치는 제외한다. 같은 파일 경로라도 영상을 교체하거나 카메라 각도가
바뀌면 기존 관측치를 재사용하지 말고 새 원본 ID로 누적해야 한다.
기존 관측치는 높이만 기록하므로 같은 높이의 다른 크롭·종횡비는 자동 검증 불가.

조회·추가 저장 모두 원본 ID 사용. 일괄 누적은 동일 원본을 한 번만 처리한다.
이미 누적된 정적 데모 영상을 반복 누적할 필요는 없다.

```sh
python scripts/refresh_moran_readings.py
```

결과 `source_meta`에 관측치 원본 ID와 프레임 높이가 맞는 관측치 개수를 기록한다.
`calibration_observations_available`은 이상치 제거 전 개수이며 정확도 점수가 아니다.
나머지 5곳은 누적 관측치가 없어 현재 프레임 기준으로 판정한다.
골목 시연에 맞는 영상 교체·실측 검증은 후속 작업이다.

## 영상의 관측 부족 차량

영상에서 마지막 프레임에 처음 등장한 차량은 `UNKNOWN`이다. `STATIONARY`와
함께 장애물 후보로 포함하고, `MOVING`으로 확인된 차량만 이동 판별을 근거로
제외한다. 기존 보행자·검출 신뢰도 필터는 유지한다. 병목 자동 탐색과 지점
직접 지정 모두 같은 규칙이다. UNKNOWN 차량 폭까지 반영하고도 여유 폭이
충분하면 PASS가 가능하다.

차량 속도는 `(마지막 frame_idx - 첫 frame_idx) × frame_interval_sec`로
경과 시간을 계산한다. 중간 프레임에서 검출이 빠져도 시간이 줄지 않는다.
관측 프레임 번호가 중복·역순이면 UNKNOWN, 프레임 간격이 0·음수·비유한 값이면
입력 오류로 처리한다. 입력 프레임은 일정한 시간 간격이라는 전제이며,
불규칙한 실시간 수신 간격은 향후 실제 타임스탬프 연동이 필요하다.

## 음수 잔여 폭 점검

폭 합산 전에 측정 행 주변에 실제 차량 마스크 픽셀이 있는지 확인한다.
회전 사각형만 겹치는 빈 모서리는 포함하지 않는다. 병목 후보 접지점이 마스크
밖이면 가장 가까운 점유 행을 사용해 차량 자체가 탐색에서 빠지는 것도 방지한다.
픽셀 허용 오차는 기존 값 유지, 음수 결과를 0으로 자르지 않는다.

도로 영역 필터 적용 전, 원본 영상·마스크 대조 결과:

- a18: 기존 병목 y≈450에서 버스 마스크는 y=459부터 존재했지만 회전 사각형이
  y≈383까지 확장돼 버스 폭을 합산했다. 수정 후 잔여 폭 -0.689m→-0.066m.
  아직 도로와 오른쪽 주차 공간 구분 및 서로 다른 깊이의 스케일 검증 필요.
- a21: 오른쪽 건물 부근을 대형버스로 오검출(confidence≈0.928).
  마스크 자체가 건물에 있어 위 수정으로 제거되지 않는다. 병목 y=254에서
  해당 객체 폭 약 7.20m가 합산되며 잔여 폭은 -3.045m. 단순 신뢰도 임계값
  상향으로 제외할 근거 없음. 도로 영역 지정과 오검출 검증이 필요하다.
- a59: 같은 빈 모서리 문제 영향. 잔여 폭 1.313m→2.001m, FAIL 유지.

## 도로 영역 밖 검출 제외

`configs/road_regions.yaml`에 a18 원본 `cctv_2`, a39 원본 `cctv_3`, a21 원본 `cctv_8`의 도로
다각형을 지정했다. 원본 화면을 보고 작성한 데모 경계이며 현장 검증 전이다.
원본 별칭을 통해 모란 위치에도 동일하게 적용한다.

- 도로 다각형과 마스크가 전혀 겹치지 않는 객체만 폭·기준 차량·이동 추적에서 제외.
- 경계에 1픽셀이라도 걸친 객체는 전체 마스크 유지. 차량 폭을 잘라 줄이지 않는다.
- `detected_objects`는 원본 검출을 유지. 재판정 파일의 `source_meta.road_region`에
  경계·해상도·원본·수동 주석 여부·버전을 기록한다.
- 설정의 원본 파일명이나 입력 해상도가 다르면 오류. 설정 파일 유실도 오류 처리.
  같은 파일명으로 내용을 교체하거나 카메라 각도가 바뀌는 경우는 자동 감지하지 못한다.
- 검출이 전부 영역 밖으로 제외되면 판정 오류를 반환한다. 과거 관측치만으로 PASS를
  만들지 않는다. 실제 도로 안 기준 차량 부족·원근 오차는 별도로 검증해야 한다.

이전 누적 관측치에는 원본 마스크가 없어 도로 밖 객체를 사후 제거할 수 없다.
ROI 적용 시 `data/calibration/{SCALE_VERSION}/{원본ID}.roi-{버전}.jsonl`만 읽고 쓰며,
새 파일이 없으면 현재 프레임만 사용한다. 기존 `{원본ID}.jsonl`은 보존한다.
경계를 변경하면 버전이 바뀌므로 해당 원본에서 다시 누적해야 한다.

도로 영역별 관측치는 a18 이미지 1장 기준 3개, a39 이미지 1장 기준 2개,
a21 영상 10프레임 기준 41개.
정적 원본을 반복 누적하면 중복이 생기므로 새 경계마다 한 번만 실행한다:

```python
from scripts.accumulate_calibration import accumulate_camera
from src import camera_registry
from src.inference.calibration_store import load_observations

for source_id in ("cctv_2", "cctv_3", "cctv_8"):
    if not load_observations(source_id):
        accumulate_camera(source_id, camera_registry.get_camera(source_id)["still_url"])
```

```sh
python scripts/review_road_regions.py
python scripts/refresh_moran_readings.py
```

검토 이미지는 `data/road_region_review/`에 저장한다. 청록선은 도로 경계,
초록 상자는 유지한 검출, 빨강 상자는 제외한 검출이다. 이미지·누적 파일은
로컬 산출물이므로 원본 미디어와 함께 각 환경에서 재생성한다.

| 카메라 | 제외 객체 | 잔여 폭 변화 | 판정 변화 |
| --- | --- | --- | --- |
| a18 | 오른쪽 도로 밖 주차 차량 | -0.066 → 0.522m | FAIL 유지 |
| a21 | 건물을 대형버스로 오검출한 객체 | -3.045 → 4.007m | FAIL → UNCERTAIN |

도로 영역 필터 적용 직후에는 나머지 10곳 수치·판정 동일, 전체 PASS 3 / FAIL 4 / UNCERTAIN 5.
a21은 캘리브레이션 오차 추정이 3.556m라 PASS가 아니다. 이 결과는 정확도
검증 완료를 뜻하지 않으며, 수동 경계 검증과 실측 기준 스케일 보정이 남아 있다.

## 스케일에서 검출 신뢰도에 따른 길이 축소 제거

스케일 식을 `차종 기준 폭 / 픽셀 폭`으로 수정했다. 이전 식은 여기에 검출
신뢰도를 곱해 동일한 1.8m 기준 차량도 신뢰도 0.5일 때 0.9m로 계산했다.
폭 3.8m 도로·폭 2.5m 소방차·여유 0.25m의 합성 사례에서 신뢰도만 낮추자
FAIL이 PASS로 바뀌는 문제가 재현됐다. 이제 두 경우 모두 장애물 폭 1.8m,
잔여 폭 2.0m, FAIL이다. 이미지·영상 및 병목 자동 탐색·지점 지정 모두 검증했다.

신뢰도 0.5 기준 차량 선택과 여러 기준자의 가중 평균은 유지한다. `local_scale()`은
이제 픽셀 폭·차종만 받는다. 픽셀 폭이 0·음수·비유한 값이면 계산 오류로 처리한다.
차종 기준 폭은 평균 가정값이므로 이 수정만으로 실제 차량 폭이 검증되는 것은 아니다.

새 관측치는 `data/calibration/vehicle_width_over_pixels_v2/`에 저장한다.
이전 식의 관측치와 혼합하거나 이전 파일로 대체하지 않는다. ROI 버전별 분리도
유지한다. 출력 `method`는 `yolov11_homography_v2` 또는
`yolov11_homography_v2_motion_aware`, 재판정 `source_meta.calibration_scale_version`은
`vehicle_width_over_pixels_v2`다.

기존 관측치가 있던 7개 원본만 재수집, 총 162개 저장. a39 원본 이미지의 이전
15개 관측치는 반복 수집분을 포함했고, 이번에는 1회 추론한 5개만 저장했다.
새 관측치와 이전 관측치의 차량 후보·위치·신뢰도가 대응함을 확인했다.
나머지 5곳은 이전과 같이 현재 프레임만 사용한다. 재현 시 원본별 한 번만 누적:

```python
from scripts.accumulate_calibration import accumulate_camera
from src import camera_registry
from src.inference.calibration_store import load_observations

for source_id in ("cctv_2", "cctv_3", "cctv_4", "cctv_5", "cctv_7", "cctv_8", "cctv_9"):
    if not load_observations(source_id):
        accumulate_camera(source_id, camera_registry.get_camera(source_id)["still_url"])
```

이후 `python scripts/refresh_moran_readings.py`로 12곳을 갱신한다.

| 카메라 | 이전 잔여 폭(m) | 수정 후(m) | 수정 후 판정 |
| --- | ---: | ---: | --- |
| a18 | 0.522 | -1.143 | FAIL |
| a39 | 0.455 | -2.588 | FAIL |
| a21 | 4.007 | 3.216 | UNCERTAIN |
| a10 | 1.634 | -0.361 | FAIL |
| a17 | 21.021 | 16.780 | PASS |
| a41 | 4.598 | 4.209 | PASS |
| a34 | 3.910 | 3.410 | UNCERTAIN |
| a54 | 4.557 | 4.280 | PASS |
| a49 | 3.683 | 2.804 | UNCERTAIN |
| a1 | 3.939 | 2.863 | UNCERTAIN |
| a59 | 2.001 | 0.549 | FAIL |
| a5 | 3.032 | 2.049 | FAIL (기존 UNCERTAIN) |

신뢰도 축소 제거 직후 PASS 3 / FAIL 5 / UNCERTAIN 4, 두 소방 차종 동일. 원본 검출·도로 폭 설정·ROI는
동일하다. 음수 잔여 폭 3곳은 원근·폭 추정 문제의 증거이며 실제 폭으로 해석하면
안 된다. 깊이별 스케일 검증과 독립적인 실측 잔여 폭 대조가 다음 과제다.

## 장애물별 깊이에서 미터로 환산

측정 행은 그 행을 막는 객체를 선택하는 데 사용하고, 각 객체의 전체 픽셀 폭은
자기 접지점의 스케일로 미터 환산한다. 이후 가로 위치가 겹친 그룹에서 가장 큰
미터 폭을 선택해 합산한다. 기존 방식은 모든 객체에 측정 행 스케일을 적용해,
다른 깊이의 가까운 차량까지 크게 계산했다.

a39의 측정 행 y=599에서 접지점 y=944인 차량은 기존 5.094m에서 3.263m로
계산된다. 아직 차종 평균 기준 폭 1.8m와 차이가 크므로 스케일 추정 자체의
오차가 남아 있다. 이것은 실측 정확도 검증 결과가 아니다.

`calibration_error_m`은 기존 측정 행 상대 오차와 포함된 객체별 상대 오차 중
큰 값을 도로 폭에 곱한다. 픽셀 폭 대신 미터 폭을 비교하면서 큰 보정 오차가
가려지지 않도록 한다. 병목 후보가 바뀌면 선택된 지점의 오차도 달라질 수 있다.
이미지·영상, 병목 자동 탐색·지점 지정에 적용한다. 영상 이동 분류는 기존 방식 유지.

출력 `method`는 `yolov11_homography_v3` 또는
`yolov11_homography_v3_motion_aware`. 관측치의 단위·산출 식은 같아 v2 저장소의
162개 관측치를 그대로 사용한다. 재수집은 필요 없다.

| 카메라 | 이전 잔여 폭(m) | 수정 후(m) | 판정 |
| --- | ---: | ---: | --- |
| a18 | -1.143 | -0.415 | FAIL 유지 |
| a39 | -2.588 | -0.228 | FAIL 유지 |
| a10 | -0.361 | 0.692 | FAIL 유지 |
| a17 | 16.780 | 22.041 | PASS → UNCERTAIN |

a17은 보정 오차 추정 23.185m 때문에 통과 확정 불가. 현재 전체 PASS 2 / FAIL 5 /
UNCERTAIN 5이며 두 소방 차종 동일. 원본 검출·도로 폭·ROI·누적 관측치 수 유지.

`python scripts/review_depth_scales.py`로 음수 폭이었던 3곳의 마지막 프레임을
대조한다. 로컬 `data/depth_scale_review.json`에 객체별 공통 스케일/자기 깊이
스케일 폭, 기준 차량 깊이 범위, 그 범위를 벗어나는 객체를 기록한다.
이 진단은 영상 이동 제외 전 결과이며 서비스 재판정은
`python scripts/refresh_moran_readings.py`로 실행한다.

남은 한계: 깊이 가중 평균 자체가 먼 기준 차량의 영향을 받고, a18 버스 접지점
y≈905는 기준 차량 범위 y≈258~450 밖이다. 단안 영상의 화면 행 겹침이 실제
지면에서 나란히 막는다는 보장도 없다. 임의 계수 조정 전에 독립적인 실측 폭·
지면 기준점이 필요하며, 음수 폭 2곳을 실제 잔여 폭으로 해석하면 안 된다.

## 실측 폭의 의미와 판정 품질 표시

사용자가 확인한 아래 값은 장애물이 없는 상태의 **건물과 건물 사이 전체 도로 폭**이다.
`wall_width_m`으로 유지하고, 잔여 폭은 `전체 도로 폭 - 장애물 점유 폭`으로 계산한다.
차량 옆 잔여 폭이나 장애물 자체의 실측 폭으로 사용하지 않는다.

| 원본 | 모란 배치 | 건물 간 전체 폭 |
| --- | --- | ---: |
| cctv_2 | a18 | 5.6m |
| cctv_3 | a39 | 7.6m |
| cctv_4 | a1 | 6.4m |
| cctv_5 | a59 | 6.0m |
| cctv_7 | a17 | 28.0m |

이 실측값을 영상 스케일의 기준으로 쓰려면 동일한 측정 구간의 양 끝점이 원본
이미지에서 어느 픽셀인지 연결해야 한다. 전체 폭 숫자만으로 특정 깊이의 m/px를
확정하지 않는다.

v4는 `ReadingCore.quality_flags`와 출력 `source_meta.measurement_quality`에
자동으로 확인한 측정 문제를 기록한다.

- `negative_effective_width`: 잔여 폭이 음수. 계산 원값과 FAIL을 보존한다.
- `outside_calibration_depth`: 장애물 접지점이 실제 사용한 기준 차량의 최소·최대
  깊이 범위 밖이다. 자동 탐색은 전체 장애물, 지점 지정은 해당 행의 장애물을 검사한다.
- `motion_scale_outside_calibration_depth`: 마지막 프레임의 장애물 차량을 MOVING으로
  제외하는 데 쓴 공통 스케일의 측정 위치가 기준 깊이 범위 밖이다.

문제가 있으면 `pass_blocked: true`로 표시하고 PASS만 UNCERTAIN으로 제한한다.
기존 FAIL은 유지한다. 범위 경계는 포함하며 float32 반올림 0.000001px만 허용한다.
보행자·낮은 검출 신뢰도·ROI 밖 객체는 기존 장애물 제외 규칙을 따른다.
영상은 이동 제외 전 검출과 병목 계산에 사용한 장애물 기준자 풀을 모두 확인한다.

`measurement_quality`는 새 계산에서 다시 생성하므로 예전 결과의 경고가 남거나
호출자가 전달한 메타데이터로 경고가 지워지지 않는다. 플래그가 없다는 것은
이 검사에서 문제가 발견되지 않았다는 뜻이며, 현장 검증 완료를 뜻하지 않는다.
기존 `confidence`는 표시용 계산값을 유지하므로 경로 선택에는 PASS 여부를 사용한다.

v4 시점 a18·a39는 음수 폭/범위 밖 계산, a1은 범위 밖 계산이 표시됐다. 12곳의 폭·판정은
동일하게 PASS 2 / FAIL 5 / UNCERTAIN 5. 출력 method는 `yolov11_homography_v4`
또는 `yolov11_homography_v4_motion_aware`이며, v2 관측치 저장소는 그대로 사용한다.

## 원본 좌표 마스크 복원 (v5)

이전 구현은 모델 입력의 letterbox 여백을 포함한 마스크를 원본 크기로 늘렸다.
이 때문에 bbox와 마스크의 좌표가 어긋나 차량 픽셀 폭·접지점·깊이별 보정이 왜곡됐다.
이제 [Ultralytics의 `retina_masks=True`](https://docs.ultralytics.com/modes/predict/)로
여백을 제거한 원본 크기 마스크를 받고, 크기나 객체 수가 맞지 않으면 측정을 중단한다.
a18의 한 차량은 마스크 세로 범위가 175~222px에서 149~199px로 복원됐다.

기존 7개 원본의 보정 관측치를 한 번씩 재수집해 총 163개 저장했다.
현재 저장소는 `data/calibration/vehicle_width_over_pixels_native_masks_v3/`이며
v1·v2 파일은 보존하되 읽지 않는다. ROI별 분리도 유지한다. 출력 method는
`yolov11_homography_v5` 또는 `yolov11_homography_v5_motion_aware`다.

12곳을 다시 추론한 결과 검출 차종·신뢰도·bbox는 모두 이전과 동일했다.
전체 판정은 두 소방차 규격 모두 PASS 2 / FAIL 5 / UNCERTAIN 5로 유지됐다.
a18 잔여 폭은 -0.415m → -0.074m, a39는 -0.228m → -0.027m로 변경됐다.
두 곳 모두 음수 폭과 보정 깊이 범위 밖 경고를 유지한다. 좌표 복원 검증은 실제
통행 가능 폭의 정확도 검증과 별개다. 실제 측정 구간·장애물 점유 폭 검증은 남아 있다.

검증: 전체 테스트 224개 통과. 원본 픽셀 보존, 잘못된 마스크 크기·개수 거부,
이전 버전 보정 관측치 미사용 회귀 검사를 포함한다.

## a39 울타리 안 주차장 제외

a39 원본은 오른쪽 울타리로 골목과 주차장이 분리돼 있지만, 주차장 검출도
장애물 폭과 보정 기준에 포함하고 있었다. 원본 화면을 확인해 `cctv_3`의 도로
다각형을 추가했다. 현장 측량 경계가 아닌 수동 영상 주석이다.

실제 원본 추론에서 골목 검출 0·2는 유지하고 주차장 검출 1·3·4·5·6은 제외했다.
경계와 겹치는 차량은 전체 마스크를 유지한다. 출력의 원본 검출 7개도 보존한다.
새 ROI 버전 `b7e0f373dfa7fe5f`에서 기준 관측치 2개를 재수집했으며, 주차장 차량이
섞인 이전 관측치 5개는 읽지 않는다. 실제 사용 중인 7개 원본의 관측치 합계는 160개다.

| a39 항목 | 이전 | 수정 후 |
| --- | ---: | ---: |
| 도로 전체 폭 | 7.600m | 7.600m |
| 추정 장애물 폭 | 7.627m | 3.791m |
| 추정 잔여 폭 | -0.027m | 3.809m |
| 보정 오차 지표 | 4.687m | 2.938m |
| 판정 (두 소방차 규격 동일) | FAIL | UNCERTAIN |

다른 11곳은 측정 시각을 제외한 모든 출력 필드가 동일하다. 전체 결과는
PASS 2 / FAIL 4 / UNCERTAIN 6. 전체 테스트 226개 통과, 원본 위 경계 시각 검토 완료.
재현: 위 도로 영역 관측치 수집 예제 → `scripts/review_road_regions.py` →
`scripts/refresh_moran_readings.py` 순서로 실행한다.

a39 영역 수정 시점에도 a18 잔여 폭 -0.074m는 남았다. 병목에 합산된 두 차량의 추정 폭은
2.876m·2.798m이며, 기준 깊이 범위는 y=239~443px다. 첫 차량의 접지점
y≈487px와 가까운 버스 검출의 접지점 y≈927px는 이 범위를 벗어난다.
도로 밖 차량을 추가로 제외할 근거가 없어 기존 FAIL·측정 경고를 유지했다.
실측 잔여 폭·동일 구간 픽셀 기준 없이 이 값을 임의 보정하지 않는다.

## 접지점 사이 병목 누락 수정 (v6)

접지점만 검사하면 비스듬한 차량의 마스크 끝과 다른 차량이 겹치는 구간을
놓칠 수 있었다. 기준 깊이 범위 안의 합성 장면에서 자동 탐색은 잔여 폭
3.20m·PASS, 누락된 y=200px를 직접 검사하면 1.418m·FAIL로 재현됐다.
이미지와 정지 차량 영상 모두 같은 오류였다.

이제 접지점에 더해 마스크의 세로 점유 구간 시작·끝과 각 구간 사이를 검사한다.
기존 세로 허용폭을 적용하므로 정수 픽셀 사이에서만 겹치는 경우도 포함한다.
마스크 내부 빈 구간도 유지한다. 전체 픽셀 행을 매번 계산하는 대신 장애물
포함 여부가 바뀌는 경계를 사용하며, 기존 폭 환산식과 보정 관측치는 유지한다.

전체 테스트 231개 통과. 기존 구현에서 실패한 회귀 사례를 수정 후 검증했고,
합성 장면의 모든 픽셀 행을 직접 검사한 최소 폭과 자동 탐색 결과도 대조했다.
출력 method는 `yolov11_homography_v6` 또는 `yolov11_homography_v6_motion_aware`다.

실제 12곳 재판정: PASS 2 / FAIL 4 / UNCERTAIN 6으로 유지.
a18은 y≈446.6px의 추가 겹침이 발견돼 추정 잔여 폭이 -0.074m → -1.913m로
변경됐다. 다른 11곳의 잔여 폭·판정과 12곳의 전체 도로 폭·원본 검출은 동일하다.
음수 폭을 줄이도록 값을 조정한 것이 아니며, a18의 스케일 정확도는 해결되지 않았다.
화면상 마스크 겹침이 지면에서 같은 통행 단면을 막는다는 보장도 없으므로,
실제 통행 폭 검증에는 동일 구간의 측정 위치·지면 기준이 여전히 필요하다.

## 추정 도로 폭으로 PASS 확정 방지 (v7)

기존 PASS인 a41·a54의 도로 폭은 모두 실측이 아닌 유사 골목 평균 6.08m였다.
7개 카메라가 `estimated_avg_of_similar_alleys` 출처를 사용하지만 판정에 반영되지
않았고, 출력 설명에는 실측 검증된 대체 영상이라고 표시돼 있었다.

이제 등록 카메라는 측정 출처(`kakao_map`, `naver_map`, `field_measurement`)와
입력 폭의 일치를 검사한다. 별칭이면 동일 영상 원본의 폭·출처도 검사한다.
추정·출처 누락/알 수 없음·미등록 ID·폭 불일치는 `unverified_wall_width`로 기록하고
PASS를 UNCERTAIN으로 제한한다. 기존 FAIL과 계산 원값은 유지한다.
카메라 ID 없는 직접 호출은 기존 계약대로 호출자가 측정 폭을 제공하는 경로다.
출처 문자열 검사는 현장 측량 정확도나 영상의 지면 보정까지 검증하지 않는다.

재판정은 12곳 모두 `wall_width_source`, `width_reference_cctv_id`,
`wall_width_scope=source_footage`를 현재 등록 정보로 덮어쓴다. 대체 영상 설명도
측정 또는 추정값을 시연에 사용한다는 문구로 교체한다. 폭 설정을 복원한 뒤에는
`scripts/refresh_moran_readings.py`를 실행해 새 등록 정보로 다시 판정한다.

- 추정 폭 7곳: a21, a10, a41, a34, a54, a49, a5.
- a41·a54: PASS → UNCERTAIN. 다른 10곳의 판정과 12곳의 폭·검출·보정 오차는 동일.
- 현재 결과: 두 소방차 규격 모두 PASS 0 / FAIL 4 / UNCERTAIN 8.
- 테스트 260개 통과. 이미지·영상, 자동/지정 지점, 원본/별칭 출처 불일치,
  기존 FAIL 보존, 오래된 출력 설명 갱신을 검증했다.

method는 `yolov11_homography_v7` 또는 `yolov11_homography_v7_motion_aware`다.
원본 도로 폭 측정이 필요한 상태이며, PASS가 생기도록 평균 폭을 조정하지 않는다.

## 최근 5개 프레임의 이동 판정 (영상 v9)

영상 10장을 1초 간격으로 추출하고 마지막 5장(6~10번째) 안에서 차량의
연속 관측 사이 이동거리를 합산해 경과 시간으로 나눈다. 평균 속도 0.3m/s 이상이면
MOVING, 미만이면 STATIONARY다. 마지막 2장이 정지해 있어도 최근 5장 구간에서
이동이 확인되면 MOVING을 유지한다. 전체 10장 평균 조건은 적용하지 않는다.
구간 중간에 움직였다가 원위치로 돌아온 경우도 이동거리 합에 반영한다.

이동 추적에는 검출 bbox 중심을 사용한다. 회전 마스크의 아래쪽 꼭짓점을 쓰면
마스크 모양 변화만으로 추적 위치가 크게 바뀔 수 있었다. 실제 a17 정지 트럭에서
접지점 x가 약 1202→1612px로 튀는 현상을 원본 두 프레임과 대조해 확인했다.
차량 폭 계산에 사용하는 마스크·접지점은 그대로 유지한다.

창은 최근 검출 5개가 아니라 입력 프레임 번호로 자른다. 검출이 누락돼도
6번째 이전의 관측을 끌어오지 않는다. 최근 구간의 관측이 2번 미만이면 UNKNOWN.
5장이 모두 잡히면 첫·마지막 사이 경과 시간은 4초이며, 누락 시에도 실제 프레임
인덱스 차를 사용한다. 5장보다 짧은 입력은 들어온 프레임 범위 안에서 계산한다.
STATIONARY와 UNKNOWN은 장애물로 유지하며 사진 입력에는 이동 판정을 적용하지 않는다.

전체 테스트 274개 통과. 최근 구간에서만 출발, 마지막 2장 정지, 최근 5장 전체
정지, 왕복 이동, 마스크 모양 변화, 검출 누락, 자동/지정 지점의 장애물 제외를 확인했다.
실제 12곳 재판정에서 a5는 잔여 폭 1.929m → 3.754m, FAIL → UNCERTAIN으로 변경됐다.
a17을 포함한 다른 11곳은 계산값·판정이 동일하다. 현재 결과는 PASS 0 / FAIL 3 / UNCERTAIN 9다.
영상 method는 `yolov11_homography_v9_motion_aware`, 이미지는 `yolov11_homography_v7`이다.

검출 박스의 위치 흔들림·추적 ID 혼동·영상 스케일 오차와 원본 7곳의 미측정 도로 폭은
별도 검증이 필요하다.

## 과거 영상의 추정 폭을 사용하는 데모 정책 (현재)

과거 CCTV 원본의 추가 실측 확보가 어려워 모란 시연은 등록된 추정 폭도 사용한다.
`scripts/refresh_moran_readings.py`는 `allow_estimated_wall_width=True`로 재판정한다.
평균 폭의 출처(`estimated_avg_of_similar_alleys`)를 실측으로 바꾸지 않으며,
`source_meta.decision_policy`에 `mode=demo_estimated_width`와 허용 여부를 기록한다.

추정 폭 때문에만 붙던 PASS 제한을 해제한다. 등록 누락·폭 불일치·알 수 없는 출처,
음수 잔여 폭·기준 깊이 범위 이탈 등은 계속 기존 방식으로 처리한다.
일반 이미지/영상 판독 함수와 배치 작업은 `allow_estimated_wall_width=False`가 기본이고,
호출자가 명시적으로 켜면 동일한 데모 정책을 적용한다. 기존 엄격한 재판정은
`refresh_readings(readings, allow_estimated_wall_width=False)`로 재현할 수 있다.

12곳의 폭·장애물 검출·보정 오차는 변경하지 않았다. a41·a54만 UNCERTAIN → PASS로
변경됐으며 전체는 두 소방차 규격 모두 PASS 2 / FAIL 3 / UNCERTAIN 7이다.
전체 테스트 288개 통과. 이 PASS는 추정값을 사용한 데모 판정이며 현장 통행 검증은 아니다.

### 다음 알고리즘 개선 방향 — 아직 미적용

1. 도로 영역 분리: 주차장·건물·보도를 분리하고 실제 통로 경계를 영상에 표시해 검토한다.
2. 지면 원근 보정: 경계선·차량 이동 궤적으로 소실점 후보를 찾고 지면 평면의 대응점을
   검증한다. 같은 화면 행에 있다는 이유로 서로 다른 깊이의 차량을 합산하는 문제를
   줄이는 것이 목표다. 근거가 부족한 장면은 임의의 변환을 적용하지 않는다.
3. 깊이 추정은 보조 후보: 차량 앞뒤 관계와 지면 평면의 일관성 검토에 사용한다.
   모델을 붙였다는 이유만으로 절대 폭이나 실제 경사각이 검증됐다고 처리하지 않는다.

[OpenCV 호모그래피 문서](https://docs.opencv.org/4.5.1/d9/dab/tutorial_homography.html)는
평면의 원근 변환을 설명한다. 평면 가정이 깨지는 경사 변화·턱은 하나의 변환으로
해결되지 않는다. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)는
상대 깊이 모델과 별도의 metric 모델을 제공한다. 현재 저장소에는 이 모델이 연결돼
있지 않다. 우선 검증 기준은 원본 12곳의 도로 경계 적합성, 같은 차량의 프레임 간
폭 안정성, a18의 깊이가 다른 차량 합산 여부다. 정답 실측이 없는 정확도 %는 제시하지 않는다.

## 실측 구간의 양 끝점 표시 도구

```sh
python scripts/prepare_measurement_review.py
```

생성된 `data/measurement_review/index.html`을 브라우저로 연다. 원본 5곳의 분석용
마지막 표본 프레임과 건물 간 전체 실측 폭을 보여준다. 프레임은 HTML에 PNG로
포함돼 서버·인터넷 연결 없이 사용할 수 있다. 영상 표본 추출은 기존 1초 간격,
최대 10장 설정을 사용하며 표본 순번·원본 파일 해시·프레임 해시를 기록한다.

1. 지면에서 실제로 측정했던 건물 간 구간의 양 끝점을 클릭한다.
2. 확대 또는 원본 픽셀 좌표 입력으로 조정한다. 조정하면 구간 일치 체크가 해제된다.
3. 해당 선이 실제 측정 구간과 일치하는지 확인한다. 확실하지 않으면 ‘위치 확인 불가’.
4. ‘주석 파일 저장’으로 `fireway-measurement-points.json`을 내려받는다.

미완료 초안도 저장·불러오기 가능하다. 브라우저 임시 저장과 별도로 JSON을 보관한다.
좌표는 확대 배율과 무관한 원본 해상도 기준이다. 초안은 `pending`, 구간 표시 및
일치 확인을 마치면 `annotated`, 위치 확인 불가면 `unavailable`로 저장된다.
`annotated`는 사용자의 구간 표시 상태이며, 기하 보정의 정확도 검증 완료 상태가 아니다.

```sh
python scripts/prepare_measurement_review.py --validate ~/Downloads/fireway-measurement-points.json
```

검증은 원본·프레임·해상도·실측 폭 일치, 카메라 중복, 프레임 밖 좌표, 중복 끝점,
미완료 상태 등을 검사한다. 화면 생성 후 원본 파일이나 실측 설정이 바뀌면 거부한다.
이 단계는 주석 준비·파일 검증이며, 기존 관측치·도로 영역·통과 판정을 변경하지 않는다.
실제 구간의 양 끝점은 위치를 아는 사람이 표시해야 한다. 이후 원근·지면 방향을
검토한 뒤 보정에 사용할 수 있다.

## 비공개 S3 연결 — 1단계

`upload_media()`는 이미지/동영상을 업로드하고 bucket/key/region, 원본 파일명,
SHA-256, Content-Type, 크기를 반환한다. `presign_media()`는 이 참조로 임시 GET URL을
발급한다(기본 900초). URL은 설정 파일에 저장하지 않는다. 로컬에서는
`profile_name="fireway"`, 배포 시에는 실행 역할의 자격 증명을 사용할 수 있다.
[AWS 공식 설명](https://docs.aws.amazon.com/boto3/latest/guide/s3-presigned-urls.html) 기준의 GET 서명 방식이다.

`register_camera.py --profile fireway`는 `s3_media`만 추가하며 기존 원본 경로,
도로 폭 출처, 캘리브레이션 별칭을 보존한다. 기존 카메라와 다른 원본/폭은 업로드 전에
거부한다. 실제 업로드·AWS 설정 변경은 아직 하지 않았다.

1단계는 저장/URL 발급 모듈이다. 아래 2단계에서 모란 재판정 실행기에 연결했다.
판정 임계값·현재 12개 결과는 이 작업에서 변경하지 않는다.

담당자 요청: [인프라](docs/handoff/infra.md) · [백엔드](docs/handoff/backend.md) ·
[프런트엔드](docs/handoff/frontend.md). 외부 저장소·인프라는 조회만 허용하며 변경하지 않는다.


## 비공개 S3 연결 — 2단계

모란 12곳의 재판정 실행기는 `--source s3`로 등록된 `s3_media`를 사용한다.
로컬 실행이 기본이다. 아래는 업로드와 실제 검증 승인 후 사용하는 명령이며 아직 실행하지 않았다.

```sh
python scripts/refresh_moran_readings.py --source s3 --profile fireway --out /tmp/moran-s3-readings.json
```

- 모든 카메라의 미디어 참조·원본 파일명·유형·해시·크기 설정을 다운로드 전에 검사한다.
- 임시 GET URL 발급 → 스트리밍 다운로드 → SHA-256/바이트 크기 일치 확인 후 디코딩한다.
- 403은 한 번 새 URL로 재시도한다. 전송 실패·해시 불일치는 전체 출력 갱신을 중단한다.
  이런 실패를 통행 가능으로 처리하거나 과거 결과로 몰래 대체하지 않는다.
- 이미지/영상은 기존 판독 함수 사용. 영상은 1초 간격 최대 10프레임, 마지막 5프레임의
  기존 움직임 판정 유지. ROI·보정 원본 ID·추정 폭 허용 정책도 그대로 전달한다.
- 성공/실패 모두 임시 파일 삭제. 서명 URL은 JSON·카메라 설정·사용자용 오류에 저장하지 않는다.
- S3 결과의 `still_public_url`은 `null`. `source_meta.s3_media`에 영속 참조,
  `original_source_url`에 보정에 쓰는 원본 식별 경로를 유지한다. 백엔드에서 화면 조회 시
  프리사인드 URL을 별도 발급해야 한다. 로컬 경로가 배포 서버에 없어도 원본 식별은 유지된다.
- 측정 불가는 이전 숫자·측정 시각 보존, UNCERTAIN 처리. 업로드된 미디어 참조는 조회 가능하며
  실패 시도 정보는 `measurement_failure`에 별도로 기록한다.

기존 `run_demo.py`의 HTTP 이미지 전용 흐름은 이번에 전환하지 않았다.
S3의 이미지·영상 혼합 재판정은 `refresh_moran_readings.py --source s3`를 사용한다.
실제 S3 네트워크·권한·브라우저 재생 검증과 백엔드 결과 수신은 아직 미완료다.

## 오프라인 인계 묶음 — 3단계

```sh
python scripts/export_moran_bundle.py
# 향후 S3 재판정 결과를 사용하려면:
python scripts/export_moran_bundle.py --readings /tmp/moran-s3-readings.json --out /tmp/moran-cctv.geojson
```

`data/moran_cctv_bundle.geojson`에 12개 실제 CCTV 점 좌표(`[경도, 위도]`),
차종별 판정, 측정 시각·출처·품질, 등록된 S3 객체 참조를 묶는다. 판정은 재계산하지 않는다.
로컬 원본 경로와 임시 미디어 URL은 내보내지 않는다. 신규 파일은 자동 생성 산출물로 Git에서 제외된다.

현재 도로 연결이 없으므로 기존의 CCTV ID를 도로 ID로 재사용하지 않고 모든 `edge_id=null`,
`edge_mapping_status=unmapped`, `routing_ready=false`로 출력한다. 목적지는 임의 지정하지 않는다.
미디어 참조가 없는 지점은 `media_status=not_registered`다. 측정 불가 지점의 숫자는
`numeric_values_status=last_known`으로 이전 측정값임을 표시한다.

이 GeoJSON은 담당자와 최종 합의할 인계 초안이며, 현재 서비스가 바로 수신하는 API 계약은 아니다.
12개 고유 ID, 좌표, 판정과 품질 제한의 충돌을 검사한 뒤 생성한다.

## 통합 검증과 실행 순서

[AI 실행 메모](docs/handoff/ai-runbook.md)에 12개 미디어 목록·등록→재판정→인계 파일 생성 순서와
실패 시 처리 범위를 정리했다. 실제 업로드는 아직 하지 않았다.
`refresh_moran_readings.py --readings`로 이전 S3 결과를 다시 입력할 수 있다.
등록 CLI는 한 건이라도 실패하면 종료 코드 1이며, 이미 성공한 개별 업로드는 보존한다.
설정·판정·GeoJSON은 완성한 임시 파일을 교체하는 방식으로 저장한다.

## 실행 환경 이동 검증

Git에는 실제 `configs/cameras.yaml`, 원본 미디어, 활성 보정 JSONL이 포함되지 않는다.
저장소만 새로 받으면 현재 12개 판정을 그대로 재현할 수 없다. 현재 상태의 이동용 묶음:

```sh
python scripts/package_moran_runtime.py --out /tmp/fireway-runtime
python scripts/package_moran_runtime.py --verify /tmp/fireway-runtime
cd /tmp/fireway-runtime
python scripts/refresh_moran_readings.py --out recomputed.json
```

기존 출력 폴더는 덮어쓰지 않는다. 코드·파인튜닝 모델·원본 12개·사용 중인 카메라와 보정 원본
설정·ROI·차량 규격·활성 보정 JSONL만 복사한다. 미사용 원본 카메라와 구버전 보정 자료는 제외.
현재 활성 보정 7파일/160관측치. 원본 경로는 묶음 기준 상대 경로로 바꾼다.
실행은 묶음 루트에서 한다. 원래 카메라 설정·판정 파일은 변경하지 않는다.

`runtime_manifest.json`에는 파일 해시와 Python/OS/주요 라이브러리 버전을 기록한다.
manifest 버전 기록 자체는 잠금 파일이 아니다. 별도의 macOS 잠금 파일은 아래 절을 참고한다.
다른 OS에서 동일 실행 결과를 보장하지 않는다.
검증 환경에서는 같은 Python 설치로 분리된 폴더에서 실제 YOLO를 실행했고, 기존 12개와
판정·폭·보정 오차·검출 결과가 모두 일치했다. 이후 같은 OS의 깨끗한 가상환경 검증도
완료했으며 아래에 기록했다. 다른 OS는 별도 검증 필요.

기본 모델 경로는 실행 디렉터리가 아닌 코드 위치 기준이다. 파인튜닝 파일이 누락되면
FileNotFoundError로 중단해 같은 이름의 기본 모델 자동 다운로드를 방지한다.
이 묶음은 원본 영상 포함 로컬 산출물이며 AWS 업로드·배포를 수행하지 않는다.


## 깨끗한 가상환경 검증 완료

검증 대상: macOS ARM64 / Python 3.13.5. 기존 시스템 site-packages를 공유하지 않는
새 venv에 패키지를 설치했다. `requirements.txt`는 주요 의존성 고정,
`requirements-lock-macos-arm64-py313.txt`는 설치된 전체 의존성의 버전 고정 파일이다.
OpenCV는 Ultralytics가 요구하는 `opencv-python` 하나만 사용해 같은 cv2 모듈을 제공하는
headless 배포판과의 중복 설치를 제거했다. GUI 기능은 사용하지 않는다.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock-macos-arm64-py313.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

새 환경에서 의존성 충돌 없음, 테스트 340개 통과. 실제 원본 12개를 실제 모델로 추론해
기존 판정·폭·오차·검출 결과 12/12 정확히 일치 확인. PASS 1 / FAIL 3 / UNCERTAIN 8 유지.
실행 묶음에도 이 잠금 파일을 포함한다. 잠금은 버전 기준이며 패키지 파일 해시를 고정한 것은 아니다.

Linux/AWS의 CPU·CUDA 환경은 별도 검증 대상이다. macOS 잠금 파일을 배포 환경용으로
검증했다고 간주하지 않는다. 실제 S3 왕복·서비스 연동도 아직 수행하지 않았다.
