"""차량 검출·분류 — YOLO11-seg + TTA (스펙 2-1장).

세그멘테이션을 쓰는 이유: 삐뚤게 주차된 차의 폭을 바운딩박스로 재면
과대추정되기 때문. 모델(yolo11n-seg.pt, Ultralytics)은 무거운 의존성이라
실제 사용 시점에 지연 임포트한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

MODEL_WEIGHTS = "yolo11n-seg.pt"  # Ultralytics, AGPL-3.0 / 상업 라이선스

# YOLO(COCO) 클래스명 → 로컬 스케일용 한글 클래스명 매핑 (스펙 2-1장 검출 클래스 확장).
# car/bus/motorcycle/truck은 캘리브레이션 기준자 + 장애물 겸용
# (goldenlane_vehicle_specs.VEHICLE_WIDTH_M 키와 맞춘다). person은 기준자가
# 아니며, 도로 폭 계산(obstacle_width_m)에서도 제외한다 — 사람은 소방차가
# 오면 스스로 비켜설 수 있어서 "차가 못 지나가게 막는" 장애물이 아니다
# (정책 확정, passable_prob.OBSTACLE_EXCLUDED_CLASSES). 검출 자체는 그대로
# detected_objects에 남는다.
#
# 매대(stand)/간판(sign)/박스(box)는 COCO 사전학습 80종에 없는 시장 특화
# 클래스라 pretrained 가중치로는 검출 불가 — 2주차 파인튜닝 대상(스펙 2-1장
# "1주차는 pretrained 그대로"). 그 전까지는 이 세 클래스가 프레임에 있어도
# 검출되지 않고 obstacle_width_m에 반영되지 않는다는 한계를 인지하고 쓸 것.
CLASS_NAME_MAP: dict[str, str] = {
    "car": "승용차",
    "bus": "버스",
    "motorcycle": "오토바이",
    "truck": "트럭",
    "person": "사람",
}

# 클래스별 mAP@0.5 (arXiv 2410.22898) — 트럭은 검출 신뢰도가 낮아 기준자로
# 쓸 때 특히 주의해야 한다는 걸 나타내는 참고값.
CLASS_MAP50: dict[str, float] = {
    "승용차": 0.837,
    "버스": 0.863,
    "오토바이": 0.679,
    "트럭": 0.355,
}


@dataclass
class VehicleDetection:
    vehicle_class: str  # 한글 차종명 (VEHICLE_WIDTH_M 키)
    confidence: float  # YOLO 분류 신뢰도
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: np.ndarray  # 세그멘테이션 마스크 (H, W), bool


_model_cache: dict[str, Any] = {}


def _load_model(weights: str = MODEL_WEIGHTS) -> Any:
    if weights not in _model_cache:
        from ultralytics import YOLO  # 무거운 의존성 — 실제 추론 시점에만 로드

        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


def detect_vehicles(frame: np.ndarray, weights: str = MODEL_WEIGHTS) -> list[VehicleDetection]:
    """세그멘테이션 기반 차량 검출. TTA(augment=True)로 노이즈를 줄인다."""
    model = _load_model(weights)
    results = model.predict(frame, augment=True, verbose=False)

    frame_h, frame_w = frame.shape[:2]
    detections: list[VehicleDetection] = []
    for result in results:
        if result.masks is None:
            continue
        names = result.names
        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()
        for i, box in enumerate(boxes):
            cls_name = names[int(box.cls[0])]
            vehicle_class = CLASS_NAME_MAP.get(cls_name)
            if vehicle_class is None:
                continue  # 차량 4종 외 검출(사람/자전거 등)은 스킵
            # YOLO의 마스크는 모델 추론 해상도(예: 384x640)로 나오고 bbox는
            # 원본 프레임 좌표계라 그대로 두면 좌표계가 어긋난다 — 폭 측정과
            # footpoint 계산이 전부 이 좌표계 위에서 이뤄지므로 원본 크기로
            # 리사이즈해서 bbox와 같은 좌표계로 맞춘다.
            mask = cv2.resize(
                masks[i].astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            detections.append(
                VehicleDetection(
                    vehicle_class=vehicle_class,
                    confidence=float(box.conf[0]),
                    bbox=tuple(box.xyxy[0].tolist()),
                    mask=mask,
                )
            )
    return detections
