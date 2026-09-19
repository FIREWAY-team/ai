"""도로 영역과 포함/제외 검출을 원본 위에 표시한다. 색상: 도로 청록, 포함 초록, 제외 빨강."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.accumulate_calibration import _load_frames_for_still_url  # noqa: E402
from src import camera_registry  # noqa: E402
from src.inference.road_region import filter_road_detections, get_road_region  # noqa: E402
from src.inference.yolo import detect_vehicles  # noqa: E402


def main():
    output = Path(__file__).resolve().parents[1] / "data/road_region_review"
    output.mkdir(parents=True, exist_ok=True)
    for source_id in ("cctv_2", "cctv_3", "cctv_8", "cctv_moran_a54"):
        region = get_road_region(source_id)
        frame = _load_frames_for_still_url(camera_registry.get_camera(source_id)["still_url"])[-1]
        detections = detect_vehicles(frame)
        kept = {id(d) for d in filter_road_detections(detections, source_id, frame.shape)}
        preview = frame.copy()
        cv2.polylines(preview, [np.array(region.polygon, dtype=np.int32)], True, (255, 255, 0), 3)
        for i, det in enumerate(detections):
            included = id(det) in kept
            color = (0, 200, 0) if included else (0, 0, 255)
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            label = f"{i} {'KEEP' if included else 'EXCLUDE'} {det.confidence:.2f}"
            cv2.putText(preview, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            print(source_id, i, det.vehicle_class, label)
        path = output / f"{source_id}.png"
        if not cv2.imwrite(str(path), preview):
            raise OSError(f"검토 이미지 저장 실패: {path}")
        print(path, region.version)


if __name__ == "__main__":
    main()
