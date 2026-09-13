"""사진 한 장 → ReadingCore 전체 오케스트레이션, 및 라우팅 모듈과의 병렬 인터페이스 (스펙 5장, v4).

판독 모듈은 카메라(사진) 한 장 단위로 완전히 독립 동작한다 — 카메라 A의
판정이 카메라 B의 판정을 참조/대기할 이유가 없는 embarrassingly parallel
구조이므로, N개 edge_id 후보를 프로세스 풀로 fan-out 처리한다. edge_id
매핑 자체는 이 모듈의 담당이 아니다 (스펙 0장) — 호출부가 jobs 리스트
순서로 결과를 자신의 edge_id에 다시 대응시킨다.

wall_width_m(벽~벽 실측 폭)은 SAM2 자동 추정 대신 카카오맵/네이버지도
거리재기로 사람이 직접 재서 카메라 등록 시 `configs/cameras.yaml`에 고정한
값을 쓴다 — 실측이 자동 추정보다 정확하고, 무거운 세그멘테이션 모델을
서버에 올릴 필요가 없어 경량 배포(AWS 프리티어)에 유리하다. 실시간
경로에는 YOLO11n-seg 하나만 남는다.

Reading.verdict 딕셔너리 안에 차종별(pump-3.5, pump-8) 판정이 이미 다
담기므로, v3에서 논의됐던 "(edge_id, vehicle_type) 쌍" 확장은 불필요하다
(스펙 5장) — FrameJudgeJob은 edge_id 단위로만 존재한다.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from goldenlane_vehicle_specs import load_vehicles_json, resolve_margin_m
from src.inference import yolo
from src.postprocess.passable_prob import build_reading_core, compute_widths
from src.schemas import ReadingCore


@dataclass
class FrameJudgeJob:
    """라우팅 모듈과의 인터페이스 계약 입력 단위 — edge_id + 프레임/캘리브레이션 정보."""

    edge_id: str
    frame: np.ndarray
    wall_width_m: float
    target_y_px: float
    camera_height_px: float
    vehicles_json: dict[str, float] | None = None
    margin_m: float | None = None


def process_frame(
    frame: np.ndarray,
    wall_width_m: float,
    target_y_px: float,
    camera_height_px: float,
    vehicles_json: dict[str, float] | None = None,
    margin_m: float | None = None,
) -> ReadingCore:
    """사진 한 장 → ReadingCore (스펙 2장 전체 파이프라인).

    wall_width_m은 카메라 등록 시 지도 실측으로 확정한 고정값
    (`configs/cameras.yaml`)을 호출부가 그대로 넘긴다.
    """
    detections = yolo.detect_vehicles(frame)

    wall_width_m, obstacle_width_m, effective_width_m, calibration_error_m = compute_widths(
        detections,
        target_y_px,
        camera_height_px,
        wall_width_m,
    )

    resolved_vehicles = vehicles_json if vehicles_json is not None else load_vehicles_json()
    resolved_margin = margin_m if margin_m is not None else resolve_margin_m()
    return build_reading_core(
        wall_width_m,
        obstacle_width_m,
        effective_width_m,
        detections,
        resolved_vehicles,
        resolved_margin,
        calibration_error_m=calibration_error_m,
    )


def _run_job(job: FrameJudgeJob) -> ReadingCore:
    return process_frame(
        job.frame,
        job.wall_width_m,
        job.target_y_px,
        job.camera_height_px,
        job.vehicles_json,
        job.margin_m,
    )


def process_camera_batch(
    jobs: list[FrameJudgeJob], max_workers: int | None = None
) -> list[ReadingCore]:
    """N개 edge_id 후보를 병렬 판정해 ReadingCore 리스트로 반환 — 네비게이션이
    한 경로 요청에 대해 후보 골목 여러 개를 동시에 판정해야 하는 경우(스펙 5장)의
    진입점이다.

    동시 처리 카메라 수(max_workers)는 배포 규모(AWS 프리티어 vCPU 수 등)에
    따라 나중에 확정한다 — 지금은 호출부가 원하는 값을 넘기거나 기본값(CPU 코어 수)에 맡긴다.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run_job, jobs))
