import numpy as np
import pytest

from scripts import motion_demo
from src import camera_registry, pipeline
from src.inference import calibration_store, yolo
from src.inference.yolo import VehicleDetection
from src.postprocess.passable_prob import compute_widths, find_narrowest_widths
from tests.helpers import make_detection, make_rotated_rect_mask


def middle_overlap_scene(height=400, lower_start=200):
    mask = make_rotated_rect_mask((height, 600), (140, 120), 40, 200, 45)
    diagonal = VehicleDetection("승용차", .9, (40, 20, 240, 220), mask)
    lower = make_detection("승용차", .9, 350, lower_start, 41, 60, (height, 600))
    return [diagonal, lower]


@pytest.mark.parametrize("video", [False, True])
def test_overlap_between_footpoints_cannot_be_reported_as_pass(monkeypatch, video):
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: {
        "coverage_test": {"wall_width_m": 5., "wall_width_source": "field_measurement"},
    })
    detections = middle_overlap_scene()
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections)
    observations = [calibration_store.make_observation(.045, .9, y, 400) for y in (50, 350)]
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: observations)
    frame = np.zeros((400, 600, 3), np.uint8)
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 5., None, 400., cctv_id="coverage_test")
    else:
        result = pipeline.process_frame(frame, 5., None, 400., cctv_id="coverage_test")
    assert result.quality_flags == []
    assert result.effective_width_m == pytest.approx(1.4180910491943362)
    assert set(result.verdict.values()) == {"FAIL"}


@pytest.mark.parametrize("holes", [False, True])
def test_automatic_width_matches_exhaustive_pixel_row_search(holes):
    detections = middle_overlap_scene()
    if holes:
        detections[0].mask[150:180] = False
        detections[1].mask[205:230] = False
    exhaustive = min(compute_widths(detections, float(y), 400., 5.)[2] for y in range(400))
    automatic = find_narrowest_widths(detections, 400., 5.)
    assert automatic[2] == pytest.approx(exhaustive)


def test_fractional_tolerance_overlap_is_not_lost_between_integer_rows():
    detections = middle_overlap_scene(height=550, lower_start=209)
    # 허용폭 2.75px: 두 마스크를 모두 포함하는 구간은 y=206.25..206.75.
    expected = compute_widths(detections, 206.5, 550., 5.)
    assert find_narrowest_widths(detections, 550., 5.)[2] == pytest.approx(expected[2])
