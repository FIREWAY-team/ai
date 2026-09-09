"""단일 카메라 모션 데모 실행 진입점 — 초 단위 연속 프레임(예: 10장)에서
정지/이동 차량을 구분해, 이동 중인 차량은 도로 폭 계산(obstacle_width_m)에서
제외한다는 걸 실제로 보여준다 (스펙 6장 "단일 카메라 데모").

로컬 디렉토리에 파일명 순서대로 저장된 연속 프레임을 읽는다 — 이 10장은
`configs/cameras.yaml`의 still_url 1장짜리 등록과는 별개 데이터라 S3에
안 올리고 로컬에서 바로 돌린다(데모 1회성 실행이라 저장소에 굳이 안 남김).

사용:
    python3 scripts/run_motion_demo.py --frames-dir data/motion_frames/cam_l1 \
        --cctv-id cam_l1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_demo import run_motion_aware_demo  # noqa: E402
from src import camera_registry  # noqa: E402
from src.adapters.common import default_target_y_px  # noqa: E402


def load_frames(frames_dir: str) -> list:
    paths = sorted(Path(frames_dir).glob("*"))
    image_paths = [p for p in paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not image_paths:
        raise ValueError(f"{frames_dir}에서 이미지 파일을 찾지 못했습니다")

    frames = []
    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
        frames.append(frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", required=True, help="연속 프레임이 파일명 순서대로 들어있는 디렉토리")
    parser.add_argument("--cctv-id", required=True, help="wall_width_m 조회용 cctv_id (configs/cameras.yaml)")
    parser.add_argument("--frame-interval-sec", type=float, default=1.0, help="프레임 간 시간 간격(초)")
    parser.add_argument("--out", help="결과를 저장할 JSON 경로 (기본: 화면 출력만)")
    args = parser.parse_args()

    frames = load_frames(args.frames_dir)
    print(f"{len(frames)}개 프레임 로드됨 ({args.frames_dir})")

    wall_width_m = camera_registry.get_camera(args.cctv_id)["wall_width_m"]
    last_frame = frames[-1]
    camera_height_px = last_frame.shape[0]
    target_y_px = default_target_y_px(last_frame)

    reading = run_motion_aware_demo(
        frames,
        wall_width_m=wall_width_m,
        target_y_px=target_y_px,
        camera_height_px=camera_height_px,
        frame_interval_sec=args.frame_interval_sec,
    )

    print(f"wall_width_m={reading.wall_width_m:.2f} obstacle_width_m={reading.obstacle_width_m:.2f} "
          f"effective_width_m={reading.effective_width_m:.2f}")
    print(f"verdict={reading.verdict} confidence={reading.confidence:.3f}")
    print("detected_objects (정지/이동 무관, 전체 검출):")
    for obj in reading.detected_objects:
        print(f"  - {obj['vehicle_class']} conf={obj['confidence']:.2f}")

    if args.out:
        payload = {
            "wall_width_m": reading.wall_width_m,
            "obstacle_width_m": reading.obstacle_width_m,
            "effective_width_m": reading.effective_width_m,
            "detected_objects": reading.detected_objects,
            "verdict": reading.verdict,
            "confidence": reading.confidence,
            "calibration_error_m": reading.calibration_error_m,
            "method": reading.method,
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장됨 -> {args.out}")


if __name__ == "__main__":
    main()
