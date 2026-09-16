"""차량 검출·분류 — YOLO11-seg + TTA (스펙 2-1장).

세그멘테이션을 쓰는 이유: 삐뚤게 주차된 차의 폭을 바운딩박스로 재면
과대추정되기 때문. 모델(yolo11n-seg.pt, Ultralytics)은 무거운 의존성이라
실제 사용 시점에 지연 임포트한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_WEIGHTS = str(Path(__file__).resolve().parents[2] / "yolo11n-seg.pt")  # Ultralytics, AGPL-3.0 / 상업 라이선스 — 파인튜닝 후에도
# 같은 파일명을 덮어써서 교체한다(경로 자체는 안 바뀜, scripts/*.py도 그대로 씀).

# 로컬 스케일용 클래스명 매핑 (스펙 2-1장 검출 클래스 확장).
#
# 2026-09-12부로 AI Hub "교통문제 해결을 위한 CCTV 교통 영상(시내도로)"(dataSetSn=165)
# Segmentation 라벨로 파인튜닝한 모델을 쓴다 — 이 모델은 COCO 영어 클래스명이 아니라
# 라벨의 한글 클래스명을 그대로 출력하므로, 이전처럼 "car"→"승용차" 식 번역이 아니라
# 그대로 통과시키는 화이트리스트 역할만 한다(모델이 우연히 다른 이름을 내면 걸러냄).
# 버스는 소형/대형을 합치지 않고 세분화 유지 — 폭이 실제로 달라서(소형버스 2.0m vs
# 대형버스 2.5m) 합치면 캘리브레이션 정확도가 떨어진다. 같은 이유로 대형 트레일러도
# 트럭과 분리 유지. 오토바이/자전거는 데이터셋 자체가 한 클래스로 묶어서 라벨링돼
# 있어 분리 불가 — "오토바이(자전거)"로 그대로 받는다.
#
# 보행자는 기준자가 아니며, 도로 폭 계산(obstacle_width_m)에서도 제외한다 — 사람은
# 소방차가 오면 스스로 비켜설 수 있어서 "차가 못 지나가게 막는" 장애물이 아니다
# (정책 확정, passable_prob.OBSTACLE_EXCLUDED_CLASSES — 여기 클래스명과 반드시
# 맞춰야 한다). 검출 자체는 그대로 detected_objects에 남는다.
CLASS_NAME_MAP: dict[str, str] = {
    "승용차": "승용차",
    "소형버스": "소형버스",
    "대형버스": "대형버스",
    "트럭": "트럭",
    "대형 트레일러": "대형 트레일러",
    "오토바이(자전거)": "오토바이(자전거)",
    "보행자": "보행자",
}

# 클래스별 mAP@0.5 — 파인튜닝 이전(COCO 기반) 참고값은 더 이상 이 모델과 무관해서
# 지웠다. 새 모델의 실제 val mAP로 채우기 전까지 비워둔다(TODO: 학습 완료 후
# runs_finetune/.../results.csv 최종 값으로 채울 것 — 특히 트럭이 이번 파인튜닝의
# 목표 클래스라 여기 수치로 개선 여부를 확인해야 한다).
CLASS_MAP50: dict[str, float] = {}


@dataclass
class VehicleDetection:
    vehicle_class: str  # 한글 차종명 (VEHICLE_WIDTH_M 키)
    confidence: float  # YOLO 분류 신뢰도
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: np.ndarray  # 세그멘테이션 마스크 (H, W), bool


_model_cache: dict[str, Any] = {}


def _load_model(weights: str = MODEL_WEIGHTS) -> Any:
    if weights not in _model_cache:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"파인튜닝 모델 파일이 없습니다: {weights}")
        from ultralytics import YOLO  # 무거운 의존성 — 실제 추론 시점에만 로드

        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def detect_vehicles(frame: np.ndarray, weights: str = MODEL_WEIGHTS) -> list[VehicleDetection]:
    """세그멘테이션 기반 차량 검출. TTA(augment=True)로 노이즈를 줄인다."""
    model = _load_model(weights)
    # 입력 letterbox 여백을 제거한 원본 좌표 마스크로 폭·접지점을 측정한다.
    results = model.predict(frame, augment=True, verbose=False, retina_masks=True)

    frame_h, frame_w = frame.shape[:2]
    detections: list[VehicleDetection] = []
    for result in results:
        if result.masks is None:
            continue
        names = result.names
        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()
        if masks.shape != (len(boxes), frame_h, frame_w):
            raise ValueError(
                f"원본 좌표 마스크가 필요합니다: expected={(len(boxes), frame_h, frame_w)}, "
                f"actual={masks.shape}"
            )
        for i, box in enumerate(boxes):
            cls_name = names[int(box.cls[0])]
            vehicle_class = CLASS_NAME_MAP.get(cls_name)
            if vehicle_class is None:
                continue  # CLASS_NAME_MAP에 없는(모델이 잘못 낸) 클래스명은 스킵
            mask = masks[i].astype(bool)
            detections.append(
                VehicleDetection(
                    vehicle_class=vehicle_class,
                    confidence=float(box.conf[0]),
                    bbox=tuple(box.xyxy[0].tolist()),
                    mask=mask,
                )
            )
    return detections
