"""단일 카메라 모션 데모 실행 진입점 — 초 단위 연속 프레임(예: 10장)에서
정지/이동 차량을 구분해, 이동 중인 차량은 도로 폭 계산(obstacle_width_m)에서
제외한다는 걸 실제로 보여준다 (스펙 6장 "단일 카메라 데모").

프레임 소스는 둘 중 하나 — 로컬 디렉토리에 파일명 순서대로 저장된 낱장
이미지들(--frames-dir), 또는 동영상 파일 하나(--video, 2026-09-15 추가:
실제 CCTV mp4에서 바로 프레임을 뽑는다 — src.adapters.video_adapter).
둘 다 `configs/cameras.yaml`의 still_url 1장짜리 등록과는 별개 데이터라
S3에 안 올리고 로컬에서 바로 돌린다(데모 1회성 실행이라 저장소에 굳이
안 남김).

출력은 file/http/rtsp 어댑터와 동일하게 build_reading()으로 감싼 팀 공용
Reading(dict) — 카메라마다 이미지로 판독하든 영상으로 판독하든
cctv_readings.json에 같은 형식으로 섞여 들어간다(2026-09-15부터. 그
전까지는 이 스크립트만 ReadingCore 필드를 따로 조합해서 형식이 달랐다).

사용:
    python3 scripts/run_motion_demo.py --frames-dir data/motion_frames/cam_l1 \
        --cctv-id cam_l1
    python3 scripts/run_motion_demo.py --video data/clips/cam_l1.mp4 \
        --cctv-id cam_l1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_demo import run_motion_aware_demo  # noqa: E402
from src import camera_registry  # noqa: E402
from src.adapters.common import build_reading, default_target_y_px  # noqa: E402
from src.adapters.video_adapter import extract_frames  # noqa: E402
from src.output.file_writer import write_readings  # noqa: E402


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="연속 프레임이 파일명 순서대로 들어있는 디렉토리")
    source.add_argument("--video", help="연속 프레임을 뽑아낼 동영상 파일(mp4 등)")
    parser.add_argument("--cctv-id", required=True, help="wall_width_m 조회용 cctv_id (configs/cameras.yaml)")
    parser.add_argument("--frame-interval-sec", type=float, default=1.0, help="프레임 간 시간 간격(초)")
    parser.add_argument("--max-frames", type=int, default=10, help="--video일 때 뽑을 최대 프레임 수")
    parser.add_argument("--out", help="결과를 저장할 JSON 경로 (기본: 화면 출력만)")
    args = parser.parse_args()

    source_path = args.video or args.frames_dir
    if args.video:
        frames = extract_frames(
            args.video, max_frames=args.max_frames, frame_interval_sec=args.frame_interval_sec
        )
        print(f"{len(frames)}개 프레임 추출됨 ({args.video})")
        source_meta = {"adapter": "video", "frame_count": len(frames)}
    else:
        frames = load_frames(args.frames_dir)
        print(f"{len(frames)}개 프레임 로드됨 ({args.frames_dir})")
        source_meta = {"adapter": "video_frames_dir", "frame_count": len(frames)}
    source_meta["frame_interval_sec"] = args.frame_interval_sec

    wall_width_m = camera_registry.get_camera(args.cctv_id)["wall_width_m"]
    last_frame = frames[-1]
    camera_height_px = last_frame.shape[0]
    target_y_px = default_target_y_px(last_frame)

    reading_core = run_motion_aware_demo(
        frames,
        wall_width_m=wall_width_m,
        target_y_px=target_y_px,
        camera_height_px=camera_height_px,
        frame_interval_sec=args.frame_interval_sec,
    )
    reading = build_reading(
        cctv_id=args.cctv_id,
        edge_id=args.cctv_id,
        still_url=source_path,
        reading_core=reading_core,
        source_meta=source_meta,
    )

    print(f"wall_width_m={reading['wall_width_m']:.2f} obstacle_width_m={reading['obstacle_width_m']:.2f} "
          f"effective_width_m={reading['effective_width_m']:.2f}")
    print(f"verdict={reading['verdict']} confidence={reading['confidence']:.3f}")
    print("detected_objects (정지/이동 무관, 전체 검출):")
    for obj in reading["detected_objects"]:
        print(f"  - {obj['vehicle_class']} conf={obj['confidence']:.2f}")

    if args.out:
        write_readings([reading], args.out)
        print(f"\n결과 저장됨 -> {args.out}")


if __name__ == "__main__":
    main()
