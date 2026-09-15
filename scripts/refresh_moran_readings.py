"""등록된 모란 원본 미디어를 현재 폭·캘리브레이션 설정으로 재판정한다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_demo import run_motion_aware_demo  # noqa: E402
from src import camera_registry  # noqa: E402
from src.adapters.common import build_reading  # noqa: E402
from src.adapters.file_adapter import load_frame  # noqa: E402
from src.adapters.video_adapter import extract_frames  # noqa: E402
from src.inference import calibration_store  # noqa: E402
from src.inference.road_region import get_road_region  # noqa: E402
from src.pipeline import process_frame  # noqa: E402

READINGS_PATH = Path(__file__).resolve().parent.parent / "data/cctv_readings_moran.json"


def refresh_readings(readings: list[dict]) -> list[dict]:
    jobs = []
    for reading in readings:
        cctv_id = reading["cctv_id"]
        camera = camera_registry.get_camera(cctv_id)
        if camera["still_url"] != reading["still_public_url"]:
            raise ValueError(f"{cctv_id}: 설정과 판정의 원본 영상이 다릅니다")
        source_id = camera_registry.calibration_source_id(cctv_id)
        adapter = reading["source_meta"]["adapter"]
        if adapter not in {"file", "video"}:
            raise ValueError(f"{cctv_id}: 지원하지 않는 원본 유형입니다")
        jobs.append((reading, camera, source_id, adapter))

    updated = []
    for reading, camera, source_id, adapter in jobs:
        cctv_id = reading["cctv_id"]
        path = camera["still_url"]
        if adapter == "video":
            frames = extract_frames(path, max_frames=10, frame_interval_sec=1.)
            height = float(frames[0].shape[0])
            core = run_motion_aware_demo(
                frames, camera["wall_width_m"], None, height, cctv_id=cctv_id,
            )
        else:
            frame = load_frame(path)
            height = float(frame.shape[0])
            core = process_frame(frame, camera["wall_width_m"], None, height, cctv_id=cctv_id)
        observations = calibration_store.load_observations(cctv_id)
        meta = {
            **reading["source_meta"],
            "calibration_source_cctv_id": source_id,
            "calibration_scale_version": calibration_store.SCALE_VERSION,
            "calibration_observations_available": sum(obs.camera_height_px == height for obs in observations),
        }
        region = get_road_region(cctv_id)
        meta.pop("road_region", None)
        if region:
            meta["road_region"] = {**region.metadata(), "version": region.version}
        updated.append(build_reading(cctv_id, reading["edge_id"], path, core, meta))
    return updated


def main() -> None:
    readings = json.loads(READINGS_PATH.read_text(encoding="utf-8"))
    updated = refresh_readings(readings)
    READINGS_PATH.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    for reading in updated:
        print(reading["cctv_id"], reading["verdict"],
              f"effective={reading['effective_width_m']:.3f}m",
              f"calibration_error={reading['calibration_error_m']:.3f}m")


if __name__ == "__main__":
    main()
