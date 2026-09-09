"""테스트용 합성 데이터 헬퍼 — 실제 YOLO/SAM2 모델 없이 순수 로직만 검증한다."""
from __future__ import annotations

import cv2
import numpy as np

from src.inference.yolo import VehicleDetection


def make_rect_mask(
    shape: tuple[int, int], x: int, y: int, w: int, h: int
) -> np.ndarray:
    """(x, y)를 좌상단으로 하는 w x h 축정렬 사각형 boolean 마스크."""
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def make_rotated_rect_mask(
    shape: tuple[int, int], center: tuple[int, int], w: int, h: int, angle_deg: float
) -> np.ndarray:
    """center를 중심으로 angle_deg만큼 회전한 w x h 사각형 boolean 마스크."""
    canvas = np.zeros(shape, dtype=np.uint8)
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(w), float(h)), angle_deg))
    cv2.fillPoly(canvas, [box.astype(np.int32)], 1)
    return canvas.astype(bool)


def make_detection(
    vehicle_class: str,
    confidence: float,
    x: int,
    y: int,
    w: int,
    h: int,
    shape: tuple[int, int] = (200, 300),
) -> VehicleDetection:
    mask = make_rect_mask(shape, x, y, w, h)
    return VehicleDetection(
        vehicle_class=vehicle_class,
        confidence=confidence,
        bbox=(x, y, x + w, y + h),
        mask=mask,
    )
