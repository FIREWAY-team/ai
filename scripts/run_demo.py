"""데모 실행 진입점 — `configs/cameras.yaml`에 등록된 카메라를 전부 판정해
`cctv_readings.json`을 만든다 (스펙 5, 6장).

네비게이션이 후보 골목 여러 개(예: 12개) 중 지나갈 수 있는 길을 골라야
하는 시나리오를 그대로 재현한다 — 카메라마다 독립적으로 판정하고
(embarrassingly parallel), 결과를 한 파일에 모은다.

사용:
    python3 scripts/run_demo.py
    python3 scripts/run_demo.py --out data/cctv_readings.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import camera_registry  # noqa: E402
from src.adapters.http_adapter import read_from_http  # noqa: E402
from src.output.file_writer import write_readings  # noqa: E402


def run_demo(out_path: str) -> list[dict]:
    cameras = camera_registry.load_cameras()
    if not cameras:
        raise RuntimeError(
            "등록된 카메라가 없습니다. 먼저 scripts/register_camera.py로 등록하세요."
        )

    readings = []
    for cctv_id, config in cameras.items():
        still_url = config.get("still_url")
        if not still_url:
            print(f"[스킵] {cctv_id}: still_url 없음 (아직 S3에 업로드 안 됨)")
            continue
        reading = read_from_http(still_url, cctv_id, edge_id=cctv_id)
        readings.append(reading)
        verdict = reading["verdict"]
        print(f"[{cctv_id}] wall={reading['wall_width_m']:.2f}m effective={reading['effective_width_m']:.2f}m verdict={verdict}")

    write_readings(readings, out_path)
    print(f"\n{len(readings)}개 카메라 판정 완료 -> {out_path}")
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="cctv_readings.json", help="출력 JSON 경로")
    args = parser.parse_args()
    run_demo(args.out)


if __name__ == "__main__":
    main()
