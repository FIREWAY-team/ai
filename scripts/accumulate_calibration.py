"""등록된 카메라들의 현재 still(또는 영상)에서 기준 차량 관측치를 뽑아
카메라별 누적 저장소(data/calibration/{SCALE_VERSION}/)에 추가한다
(정확도개선방안 A-4 확장).

CCTV는 고정 카메라라 "화면 y좌표 → 실제 깊이" 관계는 시간이 지나도 안 변한다.
그런데 한 프레임 안에 기준 차량이 1~2대뿐인 경우가 많아, 먼 지점으로 스케일을
외삽할 때 실측 검증에서 30~40%대 오차가 확인됐다(2026-09-15). 이미지 카메라는
이 스크립트를 주기 실행(cron 등)하면 그때그때 보이는 기준 차량만큼씩 쌓인다 —
다만 이 프로젝트의 데모 카메라는 정적 이미지/영상이라 실제로 주기 실행해봐야
매번 같은 내용만 다시 저장돼 cron은 의미가 없다고 판단해 뺐다(2026-09-15).

대신 still_url이 동영상(mp4 등)이면 그 영상 안에서 여러 프레임(최대
MAX_VIDEO_FRAMES개)을 뽑아 프레임마다 따로 기준 차량을 찾는다 — 정지된 채
등록된 차량은 프레임마다 거의 같은 지점만 주지만, 화면을 가로지르는 차량은
프레임마다 다른 깊이(y좌표)에서 잡혀서 한 번의 실행만으로도 이미지 카메라
한 장보다 훨씬 촘촘한 깊이별 관측치를 얻는다(cctv_4 실측 검증 중 확인:
9초 동안 화면을 가로지른 차 한 대가 footpoint_y=164→121까지 서로 다른
7개 지점을 지나갔다). 그래도 같은 저장된 영상을 다시 돌리면 완전히 같은
관측치가 또 쌓이므로(중복 제거 없음), 영상 카메라도 한 번만 실행하면
충분하고 반복 실행할 이유는 없다 — 실제 배포 후 진짜 실시간 영상이 계속
들어오는 시점부터는 다시 주기 실행이 의미 있어진다.

한 번 실행해서 기준 차량이 안 잡혀도(빈 골목 등) 손해가 없다 — 다음 실행 때
또 시도되고, 실패한 카메라 하나 때문에 나머지가 막히지 않는다.

사용:
    python3 scripts/accumulate_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import camera_registry  # noqa: E402
from src.adapters.file_adapter import load_frame  # noqa: E402
from src.adapters.http_adapter import fetch_frame  # noqa: E402
from src.adapters.video_adapter import extract_frames  # noqa: E402
from src.inference import calibration_store, yolo  # noqa: E402
from src.inference.homography import local_scale_estimates  # noqa: E402
from src.inference.road_region import filter_road_detections, get_road_region  # noqa: E402

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
MAX_VIDEO_FRAMES = 10


def _load_frames_for_still_url(still_url: str) -> list[np.ndarray]:
    """still_url이 HTTP(S)면 fetch_frame(1장), 동영상이면 extract_frames로
    여러 장, 그 외 로컬 이미지면 load_frame(1장)으로 분기한다."""
    if still_url.startswith("http://") or still_url.startswith("https://"):
        return [fetch_frame(still_url)]
    if still_url.lower().endswith(VIDEO_SUFFIXES):
        return extract_frames(still_url, max_frames=MAX_VIDEO_FRAMES)
    return [load_frame(still_url)]


def accumulate_camera(
    cctv_id: str,
    still_url: str,
    calibration_dir: Path = calibration_store.DEFAULT_CALIBRATION_DIR,
) -> int:
    """카메라 하나의 현재 still(또는 영상의 여러 프레임)에서 기준 차량
    관측치를 뽑아 누적 저장한다. 저장한 관측치 개수를 반환한다(0이면
    기준 차량이 하나도 안 잡혔다는 뜻). calibration_dir은 테스트에서
    실제 저장소를 건드리지 않도록 바꿔 끼울 수 있게 노출한다."""
    source_id = camera_registry.calibration_source_id(cctv_id)
    if source_id != cctv_id and camera_registry.get_camera(source_id)["still_url"] != still_url:
        raise ValueError(f"{cctv_id}: 다른 영상을 원본 캘리브레이션에 누적할 수 없습니다")
    if get_road_region(cctv_id) and camera_registry.get_camera(source_id)["still_url"] != still_url:
        raise ValueError(f"{cctv_id}: 도로 영역 원본이 아닌 영상을 누적할 수 없습니다")
    frames = _load_frames_for_still_url(still_url)

    observations = []
    for frame in frames:
        detections = yolo.detect_vehicles(frame)
        estimates = local_scale_estimates(filter_road_detections(detections, cctv_id, frame.shape))
        camera_height_px = frame.shape[0]
        observations.extend(
            calibration_store.make_observation(
                scale=scale,
                confidence=conf,
                footpoint_y=footpoint_y,
                camera_height_px=camera_height_px,
            )
            for scale, conf, footpoint_y in estimates
        )

    calibration_store.append_observations(cctv_id, observations, calibration_dir=calibration_dir)
    return len(observations)


def accumulate_all() -> None:
    cameras = camera_registry.load_cameras()
    if not cameras:
        raise RuntimeError(
            "등록된 카메라가 없습니다. 먼저 scripts/register_camera.py로 등록하세요."
        )

    processed_sources: set[str] = set()
    for cctv_id, config in cameras.items():
        still_url = config.get("still_url")
        if not still_url:
            print(f"[스킵] {cctv_id}: still_url 없음")
            continue
        try:
            source_id = camera_registry.calibration_source_id(cctv_id)
            if source_id in processed_sources:
                continue
            n = accumulate_camera(cctv_id, still_url)
            processed_sources.add(source_id)
        except Exception as error:  # noqa: BLE001 — 카메라 하나 실패해도 나머지는 계속
            print(f"[실패] {cctv_id}: {error}")
            continue
        print(f"[{cctv_id}] 기준 차량 {n}개 누적 저장")


if __name__ == "__main__":
    accumulate_all()
