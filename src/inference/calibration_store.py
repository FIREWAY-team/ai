"""카메라별 기준 차량 관측치 누적 저장소 (정확도개선방안 A-4 확장, 2026-09-15).

CCTV는 고정 카메라라 "화면 y좌표 → 실제 깊이" 관계는 시간이 지나도 안 변한다.
그런데 지금까지는 그때그때 프레임에 찍힌 기준 차량만으로 스케일을 계산했는데,
한 프레임 안에 기준 차량이 1~2대뿐인 경우가 많아 먼 지점으로 스케일을 외삽할
때 큰 오차(실측 검증에서 leave-one-out 평균 40%대, 벽 폭 비교 39%)가 생겼다.

이 모듈은 시간을 두고 여러 프레임에서 관측한 기준 차량을 카메라별로 누적해서,
이 카메라의 "깊이별 스케일 곡선"을 점점 촘촘하게 채우는 데 쓴다 — 알고리즘
(combine_scales_with_error)은 그대로 두고, 입력으로 넘기는 관측치 수만
늘리는 방식이라 기존 로직과 완전히 호환된다.

저장 형식은 카메라별 JSONL 파일 — 프로젝트 전체가 파일 기반이라 별도 DB
인프라 없이 사람이 읽기 쉬운 형태를 유지한다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CALIBRATION_DIR = Path(__file__).parent.parent.parent / "data" / "calibration"


@dataclass
class CalibrationObservation:
    """기준 차량 하나의 관측치. local_scale_estimates()의 (scale, conf, y)
    튜플에 관측 시각과 그 프레임의 camera_height_px를 더한 영속화 가능한 형태.
    """

    scale: float
    confidence: float
    footpoint_y: float
    camera_height_px: float
    observed_at: str  # ISO8601 (UTC)


def make_observation(
    scale: float, confidence: float, footpoint_y: float, camera_height_px: float
) -> CalibrationObservation:
    # cv2.boxPoints() 등 OpenCV 연산 결과가 섞여 들어오면 numpy.float32일 수
    # 있는데, 그대로 두면 json.dumps가 못 읽는다 — 저장 시점에 순수 파이썬
    # float로 캐스팅해서 직렬화 실패를 원천 차단한다.
    return CalibrationObservation(
        scale=float(scale),
        confidence=float(confidence),
        footpoint_y=float(footpoint_y),
        camera_height_px=float(camera_height_px),
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


def _path_for(cctv_id: str, calibration_dir: Path) -> Path:
    return calibration_dir / f"{cctv_id}.jsonl"


def append_observations(
    cctv_id: str,
    observations: list[CalibrationObservation],
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> None:
    """관측치를 카메라별 JSONL 파일에 이어 붙인다. 빈 리스트면 아무것도 안 한다
    (기준 차량이 하나도 안 잡힌 프레임에서 빈 파일을 만들지 않기 위함)."""
    if not observations:
        return
    path = _path_for(cctv_id, calibration_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obs in observations:
            f.write(json.dumps(asdict(obs), ensure_ascii=False) + "\n")


def load_observations(
    cctv_id: str,
    max_age_days: float | None = None,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> list[CalibrationObservation]:
    """누적된 관측치를 로드한다. 파일이 없으면(아직 한 번도 축적 안 된 카메라)
    빈 리스트 — 호출부는 이 경우 기존처럼 현재 프레임만으로 동작해야 한다.

    max_age_days — 카메라를 재설치/각도 변경했을 때 그 이전 관측치가 더 이상
    유효하지 않은 경우를 위한 옵션. 기본(None)은 무제한 — 카메라가 고정이면
    오래된 관측치도 여전히 유효하므로 굳이 버릴 이유가 없다.
    """
    path = _path_for(cctv_id, calibration_dir)
    if not path.exists():
        return []

    cutoff_ts: float | None = None
    if max_age_days is not None:
        cutoff_ts = datetime.now(timezone.utc).timestamp() - max_age_days * 86400

    results: list[CalibrationObservation] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if cutoff_ts is not None:
                obs_ts = datetime.fromisoformat(data["observed_at"]).timestamp()
                if obs_ts < cutoff_ts:
                    continue
            results.append(CalibrationObservation(**data))
    return results
