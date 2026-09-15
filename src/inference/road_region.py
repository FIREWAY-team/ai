"""원본 영상에 지정한 도로 영역. 경계와 겹친 객체는 전체 마스크를 유지한다."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
import yaml

from src import camera_registry
from src.inference.yolo import VehicleDetection

DEFAULT_REGIONS_PATH = Path(__file__).resolve().parents[2] / "configs/road_regions.yaml"


@dataclass(frozen=True)
class RoadRegion:
    source_id: str
    media_filename: str
    frame_size: tuple[int, int]
    polygon: tuple[tuple[int, int], ...]

    def metadata(self) -> dict:
        return {
            "source_cctv_id": self.source_id,
            "media_filename": self.media_filename,
            "frame_size": list(self.frame_size),
            "polygon": [list(point) for point in self.polygon],
            "annotation_source": "manual_visual",
            "filter_rule": "keep_any_mask_overlap_v1",
        }

    @property
    def version(self) -> str:
        payload = json.dumps(self.metadata(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        if tuple(shape[:2]) != self.frame_size[::-1]:
            raise ValueError(f"{self.source_id}: 도로 영역과 프레임 해상도 불일치")

    def mask(self) -> np.ndarray:
        mask = np.zeros(self.frame_size[::-1], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.polygon, dtype=np.int32)], 1)
        return mask.astype(bool)


def get_road_region(cctv_id: str | None) -> RoadRegion | None:
    if cctv_id is None:
        return None
    source_id = camera_registry.calibration_source_id(cctv_id)
    # 설정 파일 유실은 필터 해제로 취급하지 않는다.
    config = yaml.safe_load(DEFAULT_REGIONS_PATH.read_text(encoding="utf-8"))
    rule = config["regions"].get(source_id)
    if rule is None:
        return None
    size = rule["frame_size"]
    points = rule["polygon"]
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError(f"{source_id}: 도로 영역 해상도 오류")
    if len(points) < 3 or any(
        len(p) != 2 or any(type(v) is not int or not 0 <= v < size[i] for i, v in enumerate(p))
        for p in points
    ):
        raise ValueError(f"{source_id}: 도로 영역 꼭짓점 오류")
    if cv2.contourArea(np.array(points, dtype=np.int32)) <= 0:
        raise ValueError(f"{source_id}: 도로 영역 면적이 없습니다")
    region = RoadRegion(source_id, rule["media_filename"], tuple(size), tuple(map(tuple, points)))
    still_url = camera_registry.get_camera(source_id)["still_url"]
    if Path(unquote(urlsplit(still_url).path)).name != region.media_filename:
        raise ValueError(f"{source_id}: 도로 영역과 원본 영상 불일치")
    return region


def filter_road_detections(
    detections: Iterable[VehicleDetection],
    cctv_id: str | None,
    frame_shape: tuple[int, ...] | None = None,
) -> list[VehicleDetection]:
    detections = list(detections)
    region = get_road_region(cctv_id)
    if region is None:
        return detections
    if frame_shape is not None:
        region.validate_shape(frame_shape)
    road_mask = region.mask()
    kept = []
    for det in detections:
        region.validate_shape(det.mask.shape)
        if np.any(det.mask & road_mask):
            kept.append(det)  # 마스크를 자르면 차량 폭이 작아져 위험한 PASS가 생긴다.
    return kept
