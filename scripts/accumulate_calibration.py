"""등록된 카메라들의 현재 still에서 기준 차량 관측치를 뽑아 카메라별 누적
저장소(data/calibration/{cctv_id}.jsonl)에 추가한다 (정확도개선방안 A-4 확장).

CCTV는 고정 카메라라 "화면 y좌표 → 실제 깊이" 관계는 시간이 지나도 안 변한다.
그런데 한 프레임 안에 기준 차량이 1~2대뿐인 경우가 많아, 먼 지점으로 스케일을
외삽할 때 실측 검증에서 30~40%대 오차가 확인됐다(2026-09-15). 이 스크립트를
주기 실행(cron 등)하면, 그때그때 보이는 기준 차량만큼씩 쌓여서 이 카메라의
깊이별 스케일 곡선이 점점 촘촘해지고, 판정 정확도가 점진적으로 개선된다.

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
from src.inference import calibration_store, yolo  # noqa: E402
from src.inference.homography import local_scale_estimates  # noqa: E402


def _load_frame_for_still_url(still_url: str) -> np.ndarray:
    """still_url이 HTTP(S)면 fetch_frame, 로컬 경로면 load_frame으로 분기한다."""
    if still_url.startswith("http://") or still_url.startswith("https://"):
        return fetch_frame(still_url)
    return load_frame(still_url)


def accumulate_camera(cctv_id: str, still_url: str) -> int:
    """카메라 하나의 현재 still에서 기준 차량 관측치를 뽑아 누적 저장한다.
    저장한 관측치 개수를 반환한다(0이면 이번 프레임엔 기준 차량이 없었다는 뜻)."""
    frame = _load_frame_for_still_url(still_url)
    detections = yolo.detect_vehicles(frame)
    estimates = local_scale_estimates(detections)
    if not estimates:
        return 0

    camera_height_px = frame.shape[0]
    observations = [
        calibration_store.make_observation(
            scale=scale,
            confidence=conf,
            footpoint_y=footpoint_y,
            camera_height_px=camera_height_px,
        )
        for scale, conf, footpoint_y in estimates
    ]
    calibration_store.append_observations(cctv_id, observations)
    return len(observations)


def accumulate_all() -> None:
    cameras = camera_registry.load_cameras()
    if not cameras:
        raise RuntimeError(
            "등록된 카메라가 없습니다. 먼저 scripts/register_camera.py로 등록하세요."
        )

    for cctv_id, config in cameras.items():
        still_url = config.get("still_url")
        if not still_url:
            print(f"[스킵] {cctv_id}: still_url 없음")
            continue
        try:
            n = accumulate_camera(cctv_id, still_url)
        except Exception as error:  # noqa: BLE001 — 카메라 하나 실패해도 나머지는 계속
            print(f"[실패] {cctv_id}: {error}")
            continue
        print(f"[{cctv_id}] 기준 차량 {n}개 누적 저장")


if __name__ == "__main__":
    accumulate_all()
