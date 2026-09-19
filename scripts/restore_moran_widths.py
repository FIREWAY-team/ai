"""모란 대체 영상과 동일한 원본의 지도 측정 폭을 복원하고 해당 판정을 다시 계산한다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.motion_demo import run_motion_aware_demo  # noqa: E402
from src.adapters.common import build_reading  # noqa: E402
from src.adapters.file_adapter import read_from_file  # noqa: E402
from src.adapters.video_adapter import extract_frames  # noqa: E402
from src.camera_registry import DEFAULT_CAMERAS_YAML  # noqa: E402

MAP_WIDTH_SOURCES = {"naver_map", "kakao_map"}
DEFAULT_READINGS = Path(__file__).resolve().parent.parent / "data/cctv_readings_moran.json"


def restore_widths(cameras: dict, readings: list[dict]) -> tuple[dict, list[dict], list[str]]:
    cameras, readings = deepcopy(cameras), deepcopy(readings)
    updates = []
    for reading in readings:
        cctv_id = reading["cctv_id"]
        if not cctv_id.startswith("cctv_moran_"):
            continue
        camera = cameras[cctv_id]
        if camera["still_url"] != reading["still_public_url"]:
            raise ValueError(f"{cctv_id}: 설정과 판정의 원본 영상이 다릅니다")
        sources = [
            (source_id, source)
            for source_id, source in cameras.items()
            if not source_id.startswith("cctv_moran_")
            and source.get("still_url") == camera["still_url"]
            and source.get("wall_width_source") in MAP_WIDTH_SOURCES
        ]
        if not sources:
            continue
        if len({source["wall_width_m"] for _, source in sources}) != 1:
            raise ValueError(f"{cctv_id}: 동일 원본의 도로 폭 설정이 서로 다릅니다")
        source_id, source = sources[0]
        updates.append((reading, camera, source_id, source))

    changed = []
    for reading, camera, source_id, source in updates:
        cctv_id = reading["cctv_id"]
        width = source["wall_width_m"]
        meta = {
            **reading.get("source_meta", {}),
            "wall_width_source": source["wall_width_source"],
            "width_reference_cctv_id": source_id,
            "wall_width_scope": "source_footage",
            "footage_note": "실주소 미촬영 — 다른 위치의 원본 영상·이미지와 해당 원본의 도로 폭을 시연용으로 배치",
        }
        # 기존 생성과 동일한 프레임·모란 캘리브레이션 ID 사용. 이번 변경 변수는 도로 폭.
        if meta.get("adapter") == "video":
            frames = extract_frames(camera["still_url"], max_frames=10, frame_interval_sec=1.0)
            core = run_motion_aware_demo(
                frames, wall_width_m=width, target_y_px=None,
                camera_height_px=float(frames[0].shape[0]), cctv_id=cctv_id,
            )
            updated = build_reading(cctv_id, reading["edge_id"], camera["still_url"], core, meta)
        elif meta.get("adapter") == "file":
            updated = read_from_file(
                camera["still_url"], cctv_id=cctv_id, wall_width_m=width, edge_id=reading["edge_id"],
            )
            updated["source_meta"] = meta
        else:
            raise ValueError(f"{cctv_id}: 지원하지 않는 원본 유형입니다")
        camera.update(wall_width_m=width, wall_width_source=source["wall_width_source"])
        reading.update(updated)
        changed.append(cctv_id)
    return cameras, readings, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS_YAML)
    parser.add_argument("--readings", type=Path, default=DEFAULT_READINGS)
    args = parser.parse_args()
    registry = yaml.safe_load(args.cameras.read_text(encoding="utf-8"))
    readings = json.loads(args.readings.read_text(encoding="utf-8"))
    cameras, readings, changed = restore_widths(registry["cameras"], readings)
    # 모든 재판정이 성공한 뒤 저장. 추론 실패로 일부 카메라만 갱신하지 않는다.
    registry["cameras"] = cameras
    args.cameras.write_text(yaml.safe_dump(registry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    args.readings.write_text(json.dumps(readings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"원본 지도 측정 폭 복원: {len(changed)}개 — {', '.join(changed)}")


if __name__ == "__main__":
    main()
