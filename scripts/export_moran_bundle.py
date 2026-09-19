"""AI 판정의 모란 CCTV 점 레이어를 오프라인 인계용 GeoJSON으로 내보낸다."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adapters.s3_adapter import validate_media
from src.output.atomic_writer import write_text_atomic

ROOT = Path(__file__).resolve().parents[1]
DESTINATION_PATH = ROOT / "configs/moran_destination.json"
EXPECTED_IDS = {f"cctv_moran_a{i}" for i in (18, 39, 21, 10, 17, 41, 34, 54, 49, 1, 59, 5)}
MEASUREMENT_FIELDS = (
    "wall_width_m", "obstacle_width_m", "effective_width_m", "calibration_error_m",
    "confidence", "measured_at", "method", "verdict",
)
PROVENANCE_FIELDS = (
    "wall_width_source", "wall_width_scope", "width_reference_cctv_id",
    "calibration_source_cctv_id", "calibration_scale_version", "calibration_observations_available",
    "decision_policy", "measurement_quality", "road_region", "measurement_failure",
)
MEDIA_FIELDS = ("bucket", "key", "region", "original_filename", "sha256", "content_type", "size_bytes")


def export_bundle(readings: list[dict]) -> dict:
    """실제 도로 연결 없는 CCTV 점 레이어. 판정·폭은 재계산하지 않는다."""
    ids = [r["cctv_id"] for r in readings]
    if len(ids) != 12 or set(ids) != EXPECTED_IDS:
        raise ValueError("모란 CCTV 12개의 고유 ID가 모두 필요합니다")
    destination = json.loads(DESTINATION_PATH.read_text(encoding="utf-8"))
    for key, limit in (("lat", 90), ("lon", 180)):
        value = destination[key]
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f"잘못된 목적지 {key}")
    radius = destination["radius_m"]
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("목적지 반경은 유한한 양수여야 합니다")
    features = []
    counts: dict[str, Counter] = {}
    for reading in readings:
        cctv_id = reading["cctv_id"]
        meta = reading["source_meta"]
        lat, lon = meta["lat"], meta["lon"]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (lat, lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{cctv_id}: 잘못된 CCTV 좌표")
        status = meta.get("measurement_status")
        if status not in {"computed", "unavailable"}:
            raise ValueError(f"{cctv_id}: 측정 상태가 필요합니다")
        verdict = reading["verdict"]
        if not verdict or any(v not in {"PASS", "FAIL", "UNCERTAIN"} for v in verdict.values()):
            raise ValueError(f"{cctv_id}: 잘못된 통행 판정")
        if status == "unavailable" and any(v != "UNCERTAIN" for v in verdict.values()):
            raise ValueError(f"{cctv_id}: 측정 불가 결과는 UNCERTAIN이어야 합니다")
        if meta.get("measurement_quality", {}).get("pass_blocked") and "PASS" in verdict.values():
            raise ValueError(f"{cctv_id}: 품질 제한과 PASS가 충돌합니다")
        for vehicle, value in verdict.items():
            counts.setdefault(vehicle, Counter())[value] += 1
        source = meta.get("original_source_url") or reading.get("still_public_url")
        media = meta.get("s3_media")
        if not source:
            raise ValueError(f"{cctv_id}: 원본 식별자가 필요합니다")
        if media is not None:
            validate_media(media, source, meta["adapter"])
        a, b = math.radians(destination["lat"]), math.radians(lat)
        delta_lon = math.radians(lon - destination["lon"])
        haversine = math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(delta_lon / 2) ** 2
        distance = 6371000 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))
        properties = {
            "distance_to_destination_m": distance,
            "within_destination_radius": distance <= radius,
            "cctv_id": cctv_id, "source_csv_id": meta["source_csv_id"],
            "address": meta.get("address"),
            "edge_id": None, "edge_mapping_status": "unmapped",
            "footage_mode": "substitute_demo", "footage_note": meta.get("footage_note"),
            "original_filename": Path(unquote(urlsplit(source).path)).name,
            "media_type": {"file": "image", "video": "video"}[meta["adapter"]],
            "media_status": "registered" if media is not None else "not_registered",
            "media": {k: deepcopy(media[k]) for k in MEDIA_FIELDS} if media else None,
            "measurement_status": status,
            "numeric_values_status": "current" if status == "computed" else "last_known",
            "measurement": {k: deepcopy(reading[k]) for k in MEASUREMENT_FIELDS},
            "provenance": {k: deepcopy(meta[k]) for k in PROVENANCE_FIELDS if k in meta},
        }
        features.append({"type": "Feature", "id": cctv_id,
                         "geometry": {"type": "Point", "coordinates": [lon, lat]},
                         "properties": properties})
    bundle = {
        "type": "FeatureCollection", "schema_version": "fireway-cctv-handoff-v1",
        "routing_ready": False, "destination": destination,
        "limitations": ["도로 구간 연결 미완료", "목적지 출입구·진입점 미확인", "타 지역 시연용 원본 영상"],
        "summary": {"camera_count": 12, "nearby_camera_count": sum(f["properties"]["within_destination_radius"] for f in features), "registered_media_count": sum(f["properties"]["media"] is not None for f in features),
                    "verdict_counts": {vehicle: dict(count) for vehicle, count in counts.items()}},
        "features": features,
    }
    # 파일로 내보내기 전에 NaN 등 JSON 비표준 값도 거부한다.
    json.dumps(bundle, allow_nan=False)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, default=ROOT / "data/cctv_readings_moran.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/moran_cctv_bundle.geojson")
    args = parser.parse_args()
    bundle = export_bundle(json.loads(args.readings.read_text(encoding="utf-8")))
    write_text_atomic(args.out, json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(bundle["summary"], ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
