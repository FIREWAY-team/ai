"""음수 폭 3곳의 마지막 프레임에서 공통 스케일과 차량별 스케일을 대조한다.

관측치에서 계산한 진단값이며 실측 정확도 점수가 아니다. 영상 이동 제외 전 비교다.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.accumulate_calibration import _load_frames_for_still_url  # noqa: E402
from src import camera_registry  # noqa: E402
from src.inference.homography import (  # noqa: E402
    combine_scales_with_error, get_footpoint, scale_estimates_with_history, vehicle_pixel_width,
)
from src.inference.road_region import filter_road_detections  # noqa: E402
from src.inference.yolo import detect_vehicles  # noqa: E402
from src.postprocess.passable_prob import (  # noqa: E402
    OBSTACLE_MIN_CONFIDENCE, OBSTACLE_EXCLUDED_CLASSES,
    OBSTACLE_Y_TOLERANCE_MIN_PX, OBSTACLE_Y_TOLERANCE_RATIO,
    _mask_has_pixels_at_y, find_narrowest_widths, obstacle_widths_m,
)


def main():
    reports = []
    for suffix in ("a18", "a39", "a10"):
        cctv_id = f"cctv_moran_{suffix}"
        camera = camera_registry.get_camera(cctv_id)
        frame = _load_frames_for_still_url(camera["still_url"])[-1]
        detections = filter_road_detections(detect_vehicles(frame), cctv_id, frame.shape)
        height = frame.shape[0]
        wall, obstacle, effective, error, target = find_narrowest_widths(
            detections, height, camera["wall_width_m"], cctv_id
        )
        estimates = scale_estimates_with_history(detections, height, cctv_id)
        common_scale, _ = combine_scales_with_error(estimates, target, height)
        min_y, max_y = min(e[2] for e in estimates), max(e[2] for e in estimates)
        objects = []
        for index, det in enumerate(detections):
            width_px, rect = vehicle_pixel_width(det.mask)
            foot_y = float(get_footpoint(rect)[1])
            scale, _ = combine_scales_with_error(estimates, foot_y, height)
            objects.append({
                "index": index, "vehicle_class": det.vehicle_class,
                "confidence": det.confidence, "footpoint_y_px": foot_y,
                "outside_reference_depth": not min_y <= foot_y <= max_y,
                "blocks_target": det.confidence >= OBSTACLE_MIN_CONFIDENCE
                and det.vehicle_class not in OBSTACLE_EXCLUDED_CLASSES
                and _mask_has_pixels_at_y(det.mask, target, max(OBSTACLE_Y_TOLERANCE_MIN_PX, height * OBSTACLE_Y_TOLERANCE_RATIO)),
                "pixel_width": width_px,
                "common_scale_width_m": width_px * common_scale,
                "own_depth_width_m": width_px * scale,
            })
        report = {
            "cctv_id": cctv_id, "comparison": "last_frame_before_motion_filter",
            "target_y_px": float(target), "reference_depth_range_px": [float(min_y), float(max_y)],
            "wall_width_m": wall, "common_scale_obstacle_at_same_target_m": obstacle_widths_m(detections, common_scale, target, height),
            "own_depth_obstacle_m": obstacle, "effective_width_m": effective,
            "calibration_error_m": error, "objects": objects,
        }
        reports.append(report)
        print(cctv_id, f"effective={effective:.3f}m", f"reference_y={min_y:.1f}..{max_y:.1f}")
    output = Path(__file__).resolve().parents[1] / "data/depth_scale_review.json"
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
