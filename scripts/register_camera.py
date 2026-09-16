"""로컬 이미지/영상을 비공개 S3에 등록한다.

원본 still_url·보정 연결은 유지하고 s3_media에 객체 참조만 저장한다.
프리사인드 GET URL은 src.output.s3_uploader.presign_media로 사용 직전 발급한다.
CSV 형식: cctv_id,image_path,wall_width_m (image_path에 MP4도 가능).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# `python3 scripts/register_camera.py`로 직접 실행할 때 프로젝트 루트가
# sys.path에 없어 src/scripts 패키지를 못 찾는 문제 보정.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import camera_registry  # noqa: E402
from src.output.atomic_writer import write_text_atomic
from src.output.s3_uploader import upload_media  # noqa: E402


@dataclass
class CameraEntry:
    cctv_id: str
    image_path: str
    wall_width_m: float


def read_manifest(path: str) -> list[CameraEntry]:
    entries: list[CameraEntry] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                CameraEntry(
                    cctv_id=row["cctv_id"].strip(),
                    image_path=row["image_path"].strip(),
                    wall_width_m=float(row["wall_width_m"]),
                )
            )
    return entries


def register_one(
    entry: CameraEntry,
    bucket: str,
    region: str,
    wall_width_source: str,
    slope_risk: str,
    *,
    profile_name: str | None = None,
    registry_path: Path = camera_registry.DEFAULT_CAMERAS_YAML,
) -> str:
    import math
    import yaml

    cameras = camera_registry.load_cameras(registry_path)
    existing = cameras.get(entry.cctv_id)
    if existing:
        # URL 교체로 원본 보정·ROI 연결을 잃지 않도록 업로드 전에 검증한다.
        if Path(existing["still_url"]).resolve() != Path(entry.image_path).resolve():
            raise ValueError("등록 원본과 다른 미디어입니다. 원본 변경은 별도 검증이 필요합니다")
        if not math.isclose(existing["wall_width_m"], entry.wall_width_m):
            raise ValueError("기존 도로 폭과 다릅니다. S3 등록으로 판정 기준을 바꿀 수 없습니다")
    elif not math.isfinite(entry.wall_width_m) or entry.wall_width_m <= 0:
        raise ValueError("도로 폭은 유한한 양수여야 합니다")
    media = upload_media(entry.image_path, bucket, entry.cctv_id,
                         region_name=region, profile_name=profile_name)
    if registry_path.exists():
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    camera = data.setdefault("cameras", {}).setdefault(entry.cctv_id, {
        "still_url": entry.image_path, "wall_width_m": entry.wall_width_m,
        "wall_width_source": wall_width_source, "slope_risk": slope_risk,
    })
    camera = dict(camera)  # YAML 별칭이 같은 객체를 공유해도 다른 카메라는 갱신하지 않는다.
    data["cameras"][entry.cctv_id] = camera
    camera["s3_media"] = media
    write_text_atomic(registry_path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return f"s3://{media['bucket']}/{media['key']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=camera_registry.DEFAULT_CAMERAS_YAML)
    parser.add_argument("--profile", help="AWS 프로필 (미지정 시 기본 자격 증명 사용)")
    parser.add_argument("--bucket", required=True, help="S3 버킷 이름")
    parser.add_argument("--region", default="ap-northeast-2", help="AWS 리전 (기본: 서울)")
    parser.add_argument("--wall-width-source", default="kakao_map", help="실측 출처 (기본: kakao_map)")
    parser.add_argument("--slope-risk", default="low", help="경사 스크리닝 결과 (기본: low)")

    single = parser.add_argument_group("단일 등록")
    single.add_argument("--cctv-id")
    single.add_argument("--image")
    single.add_argument("--wall-width-m", type=float)

    parser.add_argument("--manifest", help="CSV: cctv_id,image_path,wall_width_m — 여러 카메라 일괄 등록")

    args = parser.parse_args()

    if args.manifest:
        entries = read_manifest(args.manifest)
    elif args.cctv_id and args.image and args.wall_width_m is not None:
        entries = [CameraEntry(args.cctv_id, args.image, args.wall_width_m)]
    else:
        parser.error("--manifest 또는 (--cctv-id --image --wall-width-m) 조합이 필요합니다")
        return

    failed = False
    for entry in entries:
        try:
            still_url = register_one(
                entry, args.bucket, args.region, args.wall_width_source, args.slope_risk,
                profile_name=args.profile, registry_path=args.registry,
            )
        except Exception as error:  # noqa: BLE001 — CLI 진입점, 어느 카메라에서 실패했는지 알려야 함
            failed = True
            print(f"[실패] {entry.cctv_id}: {error}", file=sys.stderr)
            continue
        print(f"[등록됨] {entry.cctv_id} -> {still_url} (wall_width_m={entry.wall_width_m})")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
