"""실측 건물 간 폭의 영상 양 끝점을 표시하는 로컬 화면 생성·주석 검증."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.accumulate_calibration import _load_frames_for_still_url  # noqa: E402
from src import camera_registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/measurement_review"
SOURCES = (
    ("cctv_2", "cctv_moran_a18", "은행동 652", 5.6),
    ("cctv_3", "cctv_moran_a39", "성남동 3506", 7.6),
    ("cctv_4", "cctv_moran_a1", "상대원3동 2327", 6.4),
    ("cctv_5", "cctv_moran_a59", "상대원3동 2327", 6.0),
    ("cctv_7", "cctv_moran_a17", "상대원2동 4738-5", 28.0),
)
IDENTITY_FIELDS = ("source_cctv_id", "moran_cctv_id", "media_filename", "media_sha256",
                   "frame_sha256", "frame_size", "sample_index", "wall_width_m")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(payload: dict, manifest: dict) -> list[dict]:
    """원본 식별·좌표 형식 검증. 실제 지면 구간 일치 여부는 사람의 검토가 필요하다."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("지원하지 않는 주석 형식입니다")
    entries = payload.get("annotations")
    expected = {item["source_cctv_id"]: item for item in manifest["cameras"]}
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("카메라 주석 개수가 원본 목록과 다릅니다")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("카메라 주석 형식 오류")
        source = entry.get("source_cctv_id")
        if not isinstance(source, str) or source not in expected or source in seen:
            raise ValueError("미등록 또는 중복 카메라 주석입니다")
        seen.add(source)
        if any(entry.get(key) != expected[source][key] for key in IDENTITY_FIELDS):
            raise ValueError(f"{source}: 원본·프레임·실측 폭 식별값 불일치")
        points = entry.get("points")
        if not isinstance(points, list) or len(points) > 2:
            raise ValueError(f"{source}: 양 끝점은 최대 2개입니다")
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(
                type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value < entry["frame_size"][axis]
                for axis, value in enumerate(point)
            ):
                raise ValueError(f"{source}: 프레임 밖 또는 잘못된 픽셀 좌표입니다")
        if len(points) == 2 and math.dist(*points) < 1:
            raise ValueError(f"{source}: 양 끝점이 같거나 1px 미만입니다")
        confirmed = entry.get("matches_measured_segment")
        status = entry.get("status")
        if type(confirmed) is not bool or not isinstance(status, str) or status not in {"pending", "annotated", "unavailable"}:
            raise ValueError(f"{source}: 주석 상태 오류")
        if status == "annotated" and (len(points) != 2 or not confirmed):
            raise ValueError(f"{source}: 실측 구간 확인과 양 끝점이 필요합니다")
        if status != "annotated" and confirmed:
            raise ValueError(f"{source}: 미완료 주석은 확인 상태일 수 없습니다")
        if status == "unavailable" and points:
            raise ValueError(f"{source}: 위치 확인 불가 주석에 좌표가 있습니다")
    return entries


def prepare(output: Path = DEFAULT_OUTPUT) -> Path:
    cameras = []
    for source, alias, address, width in SOURCES:
        config = camera_registry.get_camera(source)
        if camera_registry.calibration_source_id(alias) != source or config["wall_width_m"] != width:
            raise ValueError(f"{source}: 사용자 확인한 원본·실측 폭 설정 불일치")
        if camera_registry.get_camera(alias)["wall_width_m"] != width:
            raise ValueError(f"{alias}: 원본과 실측 폭 불일치")
        path = Path(config["still_url"])
        frames = _load_frames_for_still_url(str(path))
        frame = frames[-1]
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError(f"{source}: 프레임 이미지 생성 실패")
        image_bytes = png.tobytes()
        cameras.append({
            "source_cctv_id": source, "moran_cctv_id": alias, "address": address,
            "wall_width_m": width, "media_filename": path.name,
            "media_sha256": file_digest(path), "frame_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])], "sample_index": len(frames) - 1,
            "image_data": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        })
    manifest = {"schema_version": 1, "cameras": cameras}
    template = (ROOT / "scripts/templates/measurement_review.html").read_text(encoding="utf-8")
    # 사용자 표시 문자열이 script 태그를 닫지 못하도록 JSON의 '<'를 이스케이프한다.
    html = template.replace("__MANIFEST_JSON__", json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c"))
    manifest["cameras"] = [{key: value for key, value in item.items() if key != "image_data"} for item in cameras]
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    destination = output / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", type=Path, help="저장한 주석 JSON 검증")
    args = parser.parse_args()
    if args.validate:
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["cameras"]:
            config = camera_registry.get_camera(item["source_cctv_id"])
            alias = camera_registry.get_camera(item["moran_cctv_id"])
            if (file_digest(Path(config["still_url"])) != item["media_sha256"]
                    or config["wall_width_m"] != item["wall_width_m"]
                    or alias["wall_width_m"] != item["wall_width_m"]
                    or camera_registry.calibration_source_id(item["moran_cctv_id"]) != item["source_cctv_id"]):
                raise ValueError("화면 생성 이후 원본 또는 실측 설정이 변경됐습니다. 화면 재생성 필요")
        entries = validate_submission(json.loads(args.validate.read_text(encoding="utf-8")), manifest)
        completed = sum(item["status"] == "annotated" for item in entries)
        print(f"파일 검증 통과: 구간 표시 {completed}/{len(entries)}. 기하 보정·판정 적용 전 검토용 주석입니다.")
    else:
        print(prepare(args.output))


if __name__ == "__main__":
    main()
