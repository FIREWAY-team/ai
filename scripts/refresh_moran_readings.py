"""등록된 모란 원본 미디어를 현재 폭·캘리브레이션 설정으로 재판정한다."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_demo import run_motion_aware_demo  # noqa: E402
from src import camera_registry  # noqa: E402
from src.adapters.common import build_reading  # noqa: E402
from src.adapters.file_adapter import load_frame  # noqa: E402
from src.adapters.s3_adapter import downloaded_media, validate_media
from src.adapters.video_adapter import extract_frames  # noqa: E402
from src.inference import calibration_store  # noqa: E402
from src.inference.road_region import get_road_region  # noqa: E402
from src.output.atomic_writer import write_text_atomic
from src.schemas import MeasurementUnavailableError
from src.pipeline import process_frame  # noqa: E402

READINGS_PATH = Path(__file__).resolve().parent.parent / "data/cctv_readings_moran.json"


def refresh_readings(readings: list[dict], *, allow_estimated_wall_width: bool = True,
                     use_s3: bool = False, profile_name: str | None = None) -> list[dict]:
    """모란 데모는 등록된 추정 폭 허용. 출처와 적용 정책은 출력에 보존한다."""
    jobs = []
    for reading in readings:
        cctv_id = reading["cctv_id"]
        camera = camera_registry.get_camera(cctv_id)
        if camera["still_url"] != reading["source_meta"].get("original_source_url", reading["still_public_url"]):
            raise ValueError(f"{cctv_id}: 설정과 판정의 원본 영상이 다릅니다")
        source_id = camera_registry.calibration_source_id(cctv_id)
        adapter = reading["source_meta"]["adapter"]
        if adapter not in {"file", "video"}:
            raise ValueError(f"{cctv_id}: 지원하지 않는 원본 유형입니다")
        if use_s3:
            validate_media(camera.get("s3_media", {}), camera["still_url"], adapter)
        jobs.append((reading, camera, source_id, adapter))

    updated = []
    for reading, camera, source_id, adapter in jobs:
        cctv_id = reading["cctv_id"]
        path = camera["still_url"]
        region = get_road_region(cctv_id)
        policy = {
            "mode": "demo_estimated_width" if allow_estimated_wall_width else "measured_width_required",
            "allow_estimated_wall_width": allow_estimated_wall_width,
        }
        try:
            source = downloaded_media(camera["s3_media"], path, adapter, profile_name=profile_name) if use_s3 else nullcontext(path)
            with source as input_path:
                if adapter == "video":
                    frames = extract_frames(input_path, max_frames=10, frame_interval_sec=1.)
                    height = float(frames[0].shape[0])
                    core = run_motion_aware_demo(
                        frames, camera["wall_width_m"], None, height, cctv_id=cctv_id,
                        allow_estimated_wall_width=allow_estimated_wall_width,
                    )
                else:
                    frame = load_frame(input_path)
                    height = float(frame.shape[0])
                    core = process_frame(
                        frame, camera["wall_width_m"], None, height, cctv_id=cctv_id,
                        allow_estimated_wall_width=allow_estimated_wall_width,
                    )
        except MeasurementUnavailableError as exc:
            previous = deepcopy(reading)
            meta = previous["source_meta"]
            meta["measurement_status"] = "unavailable"
            meta["measurement_failure"] = {
                "code": exc.code, "message": str(exc),
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "attempted_road_region": {**region.metadata(), "version": region.version} if region else None,
                "decision_policy": policy,
                "input_transport": "s3_presigned" if use_s3 else "local",
                "attempted_s3_media": deepcopy(camera.get("s3_media")) if use_s3 else None,
            }
            flags = meta.get("measurement_quality", {}).get("flags", [])
            meta["measurement_quality"] = {"flags": sorted(set(flags + [exc.code])), "pass_blocked": True}
            previous["verdict"] = {vehicle: "UNCERTAIN" for vehicle in reading["verdict"]}
            previous["confidence"] = 0.0
            if use_s3:
                previous["still_public_url"] = None
                meta["original_source_url"] = path
                meta["s3_media"] = deepcopy(camera["s3_media"])
            updated.append(previous)
            continue
        observations = calibration_store.load_observations(cctv_id)
        meta = {
            **reading["source_meta"],
            "wall_width_source": camera.get("wall_width_source", "unknown"),
            "width_reference_cctv_id": source_id,
            "wall_width_scope": "source_footage",
            "decision_policy": {
                "mode": "demo_estimated_width" if allow_estimated_wall_width else "measured_width_required",
                "allow_estimated_wall_width": allow_estimated_wall_width,
            },
            "footage_note": "실주소 미촬영 — 다른 위치의 원본 영상·이미지와 해당 원본의 도로 폭(측정 또는 추정)을 시연용으로 배치",
            "calibration_source_cctv_id": source_id,
            "calibration_scale_version": calibration_store.SCALE_VERSION,
            "calibration_observations_available": sum(obs.camera_height_px == height for obs in observations),
        }
        meta["input_transport"] = "s3_presigned" if use_s3 else "local"
        meta["original_source_url"] = path
        meta.pop("s3_media", None)
        if use_s3:
            meta["s3_media"] = deepcopy(camera["s3_media"])
        region = get_road_region(cctv_id)
        meta.pop("measurement_failure", None)
        meta["measurement_status"] = "computed"
        meta.pop("road_region", None)
        if region:
            meta["road_region"] = {**region.metadata(), "version": region.version}
        updated.append(build_reading(cctv_id, reading["edge_id"], None if use_s3 else path, core, meta))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=READINGS_PATH)
    parser.add_argument("--source", choices=("local", "s3"), default="local")
    parser.add_argument("--profile", help="S3 서명에 사용할 AWS 프로필")
    parser.add_argument("--out", type=Path, default=READINGS_PATH)
    args = parser.parse_args()
    readings = json.loads(args.readings.read_text(encoding="utf-8"))
    updated = refresh_readings(readings, use_s3=args.source == "s3", profile_name=args.profile)
    write_text_atomic(args.out, json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False))
    for reading in updated:
        if reading["source_meta"].get("measurement_status") == "unavailable":
            print(reading["cctv_id"], "UNCERTAIN / 측정 불가", reading["source_meta"]["measurement_failure"]["code"],
                  "이전 측정:", reading["measured_at"])
            continue
        print(reading["cctv_id"], reading["verdict"],
              f"effective={reading['effective_width_m']:.3f}m",
              f"calibration_error={reading['calibration_error_m']:.3f}m")


if __name__ == "__main__":
    main()
