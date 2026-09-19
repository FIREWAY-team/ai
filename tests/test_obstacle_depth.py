import numpy as np
import pytest

from scripts import motion_demo
from src import pipeline
from src.inference import yolo
from src.postprocess import passable_prob
from tests.helpers import make_detection


def scene():
    # 두 장애물은 y=100, 200을 모두 점유한다. 먼 기준 차량은 두 행을 막지 않는다.
    return [
        make_detection("승용차", 1., 10, 20, 91, 400, shape=(500, 600)),
        make_detection("승용차", 1., 250, 20, 46, 200, shape=(500, 600)),
        make_detection("승용차", 1., 400, 10, 20, 50, shape=(500, 600)),
    ]


@pytest.mark.parametrize("video", [False, True])
def test_same_blocking_vehicles_keep_width_when_measurement_row_changes(monkeypatch, video):
    detections = scene()
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections)
    frame = np.zeros((500, 600, 3), np.uint8)
    results = []
    for target in (100., 200.):
        if video:
            result = motion_demo.run_motion_aware_demo([frame, frame], 8., target, 500.)
        else:
            result = pipeline.process_frame(frame, 8., target, 500.)
        results.append(result)
    assert results[0].obstacle_width_m == pytest.approx(results[1].obstacle_width_m)
    assert results[0].effective_width_m == pytest.approx(results[1].effective_width_m)
    assert len(results[0].detected_objects) == 3


@pytest.mark.parametrize("overlap, expected", [(True, 4.), (False, 5.8)])
def test_vehicle_widths_converted_before_selecting_cluster_maximum(monkeypatch, overlap, expected):
    near = make_detection("승용차", .9, 10, 20, 181, 400, shape=(500, 600))
    far = make_detection("트럭", .9, 50 if overlap else 300, 20, 101, 200, shape=(500, 600))
    # 180px*0.01=1.8m, 100px*0.04=4m. 픽셀 크기 순서와 실제 폭 순서는 반대.
    def known_calibration(estimates, target, height):
        return (.01, .005) if target == 419. else (.04, .002)

    monkeypatch.setattr(passable_prob, "combine_scales_with_error", known_calibration)
    width, relative_error = passable_prob.calibrated_obstacle_widths([near, far], [], 100., 500.)
    assert width == pytest.approx(expected)
    # 폭 최대값에서 제외된 차량의 큰 보정 오차도 숨기지 않는다.
    assert relative_error == pytest.approx(.5)


def test_no_reference_at_obstacle_depth_is_not_silently_omitted(monkeypatch):
    monkeypatch.setattr(passable_prob, "combine_scales_with_error", lambda *args: (None, 0.))
    with pytest.raises(ValueError, match="접지점"):
        passable_prob.calibrated_obstacle_widths(scene(), [], 100., 500.)


def test_no_blocking_masks_means_zero_width_and_zero_local_error():
    assert passable_prob.calibrated_obstacle_widths(scene(), [], 490., 500.) == (0., 0.)


@pytest.mark.parametrize("target_error, expected", [(.004, 2.), (.02, 4.)])
def test_width_calculation_keeps_larger_target_or_obstacle_error(monkeypatch, target_error, expected):
    det = make_detection("승용차", .9, 10, 20, 91, 400, shape=(500, 600))
    def calibration(estimates, target, height):
        return (.02, .01) if target == 419. else (.02, target_error)

    monkeypatch.setattr(passable_prob, "combine_scales_with_error", calibration)
    result = passable_prob.compute_widths([det], 100., 500., 4.)
    assert result[3] == pytest.approx(expected)
