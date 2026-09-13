"""카메라(CCTV) 1대 또는 여러 대를 등록한다 — 로컬 이미지 → S3 업로드 →
`configs/cameras.yaml`에 cctv_id/wall_width_m/still_url 기록 (스펙 3장).

wall_width_m은 카카오맵/네이버지도 "거리재기"로 직접 잰 값을 그대로 넣는다
(SAM2 자동 추정은 쓰지 않음 — src/pipeline.py 참고).

단일 등록:
    python3 scripts/register_camera.py --bucket fireway-cctv-demo \
        --cctv-id cam_l1 --image ~/photos/cam_l1.jpg --wall-width-m 4.2

일괄 등록(12개 CCTV를 한 번에) — manifest CSV 형식: cctv_id,image_path,wall_width_m
    python3 scripts/register_camera.py --bucket fireway-cctv-demo \
        --manifest data/camera_manifest.csv
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
from src.output.s3_uploader import upload_image  # noqa: E402


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
) -> str:
    still_url = upload_image(
        entry.image_path, bucket, entry.cctv_id, region_name=region
    )
    camera_registry.register_camera(
        cctv_id=entry.cctv_id,
        wall_width_m=entry.wall_width_m,
        still_url=still_url,
        wall_width_source=wall_width_source,
        slope_risk=slope_risk,
    )
    return still_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    for entry in entries:
        try:
            still_url = register_one(
                entry, args.bucket, args.region, args.wall_width_source, args.slope_risk
            )
        except Exception as error:  # noqa: BLE001 — CLI 진입점, 어느 카메라에서 실패했는지 알려야 함
            print(f"[실패] {entry.cctv_id}: {error}", file=sys.stderr)
            continue
        print(f"[등록됨] {entry.cctv_id} -> {still_url} (wall_width_m={entry.wall_width_m})")


if __name__ == "__main__":
    main()
