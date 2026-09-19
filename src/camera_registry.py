"""카메라 등록소 — cctv_id ↔ wall_width_m(지도 실측값) ↔ still_url(S3) 매핑
(스펙 3장, 경량화 이후).

실측값은 이미지가 아니라 카메라(cctv_id)에 묶는다: CCTV가 고정형이면 같은
카메라에서 나온 이미지는 몇 장이든 항상 같은 도로 폭을 가지므로, 이미지
파일명이 무엇이든(UUID든 뭐든) cctv_id만 알면 wall_width_m을 찾을 수 있다.
`configs/cameras.yaml`이 이 매핑의 단일 진실 소스(source of truth)다.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CAMERAS_YAML = Path(__file__).parent.parent / "configs" / "cameras.yaml"
MEASURED_WIDTH_SOURCES = frozenset({"kakao_map", "naver_map", "field_measurement"})


def _load_raw(path: Path = DEFAULT_CAMERAS_YAML) -> dict[str, Any]:
    if not path.exists():
        return {"cameras": {}}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {"cameras": {}}


def load_cameras(path: Path = DEFAULT_CAMERAS_YAML) -> dict[str, dict[str, Any]]:
    """등록된 카메라 전체를 {cctv_id: {wall_width_m, still_url, ...}} 형태로 반환."""
    return _load_raw(path).get("cameras", {}) or {}


def get_camera(cctv_id: str, path: Path = DEFAULT_CAMERAS_YAML) -> dict[str, Any]:
    """cctv_id로 카메라 설정(wall_width_m, still_url 등)을 조회한다.

    등록 안 된 cctv_id면 KeyError — 판정 파이프라인이 잘못된 카메라로 조용히
    진행하는 것보다, 등록 누락을 그 자리에서 드러내는 쪽이 안전하다(스펙 4장
    오류 비대칭성과 같은 이유 — 조용한 fallback은 위험한 오판을 낳는다).
    """
    cameras = load_cameras(path)
    if cctv_id not in cameras:
        raise KeyError(f"등록되지 않은 cctv_id입니다: {cctv_id} (configs/cameras.yaml 확인)")
    return cameras[cctv_id]


def calibration_source_id(cctv_id: str, path: Path = DEFAULT_CAMERAS_YAML) -> str:
    """동일 원본 영상을 배치한 데모 카메라는 원본 관측치 저장소를 공유한다."""
    cameras = load_cameras(path)
    camera = cameras.get(cctv_id, {})
    source_id = camera.get("calibration_source_cctv_id", cctv_id)
    if source_id == cctv_id:
        return cctv_id
    source = cameras.get(source_id)
    if source is None:
        raise ValueError(f"{cctv_id}: 캘리브레이션 원본 {source_id} 미등록")
    if source.get("calibration_source_cctv_id", source_id) != source_id:
        raise ValueError(f"{cctv_id}: 캘리브레이션 원본은 다른 별칭을 참조할 수 없습니다")
    if not camera.get("still_url") or camera["still_url"] != source.get("still_url"):
        raise ValueError(f"{cctv_id}: 캘리브레이션 원본 영상 불일치")
    return source_id


def wall_width_quality_flags(
    cctv_id: str | None,
    wall_width_m: float,
    path: Path = DEFAULT_CAMERAS_YAML,
    *,
    allow_estimated_wall_width: bool = False,
) -> list[str]:
    """등록 카메라는 측정 출처·입력 폭·동일 영상 원본의 폭이 모두 일치해야 한다.

    ID 없는 직접 호출은 기존 계약대로 호출자가 측정 폭을 제공한다.
    데모에서는 명시적으로 등록된 유사 골목 평균 폭도 사용할 수 있다.
    """
    if cctv_id is None:
        return []
    cameras = load_cameras(path)
    source_id = calibration_source_id(cctv_id, path)
    allowed_sources = MEASURED_WIDTH_SOURCES
    if allow_estimated_wall_width:
        allowed_sources = allowed_sources | {"estimated_avg_of_similar_alleys"}
    for camera_id in {cctv_id, source_id}:
        camera = cameras.get(camera_id, {})
        width = camera.get("wall_width_m")
        if (
            camera.get("wall_width_source") not in allowed_sources
            or not isinstance(width, (int, float))
            or not math.isfinite(width) or width <= 0
            or not math.isclose(width, wall_width_m, rel_tol=1e-9, abs_tol=1e-9)
        ):
            return ["unverified_wall_width"]
    return []


def register_camera(
    cctv_id: str,
    wall_width_m: float,
    still_url: str,
    wall_width_source: str = "kakao_map",
    slope_risk: str = "low",
    path: Path = DEFAULT_CAMERAS_YAML,
    lat: float | None = None,
    lon: float | None = None,
) -> None:
    """카메라 하나를 등록/갱신한다 — `scripts/register_camera.py`가 S3 업로드
    직후 이 함수를 호출해 `configs/cameras.yaml`에 반영한다.

    lat/lon — 카메라 실제 설치 위경도(WGS84). 지금까지는 이 값이 정식
    필드가 아니라 `Reading.source_meta`에만 임시로 끼워넣어져서(모란 12곳
    등록 스크립트), no_go_areas 같은 실제 공간 데이터와 SQL로 조인할
    방법이 없었다. 둘 다 없거나 둘 다 있어야 한다 — 한쪽만 있으면 좌표가
    아니라 임의의 반쪽 데이터가 조용히 저장된다.
    """
    if (lat is None) != (lon is None):
        raise ValueError("lat/lon은 둘 다 주거나 둘 다 생략해야 합니다")
    if lat is not None and not (-90 <= lat <= 90):
        raise ValueError(f"lat 범위 초과(-90~90): {lat}")
    if lon is not None and not (-180 <= lon <= 180):
        raise ValueError(f"lon 범위 초과(-180~180): {lon}")

    data = _load_raw(path)
    cameras = data.setdefault("cameras", {})
    camera = {
        "slope_risk": slope_risk,
        "wall_width_m": wall_width_m,
        "wall_width_source": wall_width_source,
        "still_url": still_url,
    }
    if lat is not None:
        camera["lat"] = lat
        camera["lon"] = lon
    cameras[cctv_id] = camera
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
