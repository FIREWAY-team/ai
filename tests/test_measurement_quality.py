import numpy as np
import pytest

from scripts import motion_demo
from src import pipeline
from src.adapters.common import build_reading
from src.inference import calibration_store, yolo
from src.inference.homography import depth_is_supported
from src.inference.yolo import VehicleDetection
from src.postprocess.passable_prob import build_reading_core, depth_quality_flags
from tests.helpers import make_detection, make_rotated_rect_mask


@pytest.mark.parametrize("depth, supported", [(99., False), (100., True), (150., True), (200., True), (201., False)])
def test_depth_coverage_includes_endpoints_without_extrapolation(depth, supported):
    assert depth_is_supported([(.02, .9, 100.), (.01, .9, 200.)], depth) is supported


def test_one_reference_does_not_establish_coverage_elsewhere():
    assert depth_is_supported([(.02, .9, 100.)], 100.)
    assert not depth_is_supported([(.02, .9, 100.)], 101.)
    assert not depth_is_supported([], 100.)
    assert not depth_is_supported([(float("nan"), .9, 100.)], 100.)


def scene():
    return [
        make_detection("승용차", .9, 400, 10, 91, 100, shape=(500, 600)),
        VehicleDetection("승용차", .9, (70, 250, 180, 350),
                         make_rotated_rect_mask((500, 600), (120, 300), 40, 90, 45)),
    ]


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("target", [None, 315.])
@pytest.mark.parametrize("history", [False, True])
def test_outside_depth_blocks_pass_and_in_range_history_restores_eligibility(monkeypatch, video, target, history):
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: scene())
    observations = [calibration_store.make_observation(.02, .9, 400., 500.)] if history else []
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: observations)
    frame = np.zeros((500, 600, 3), np.uint8)
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 8., target, 500., cctv_id="quality_test")
    else:
        result = pipeline.process_frame(frame, 8., target, 500., cctv_id="quality_test")
    assert result.effective_width_m > 5.
    assert set(result.verdict.values()) == ({"PASS"} if history else {"UNCERTAIN"})
    assert result.quality_flags == ([] if history else ["outside_calibration_depth"])
    assert len(result.detected_objects) == 2


def test_fixed_target_does_not_use_unrelated_out_of_range_obstacle():
    estimates = [(.02, .9, 109.)]
    assert depth_quality_flags(scene(), estimates, 500., 100.) == []
    assert depth_quality_flags(scene(), estimates, 500.) == ["outside_calibration_depth"]


def test_non_blocking_classes_and_low_confidence_do_not_trigger_depth_flag():
    dets = [make_detection("보행자", .9, 10, 10, 30, 100), make_detection("승용차", .3, 100, 10, 30, 100)]
    assert depth_quality_flags(dets, [], 200.) == []


def test_negative_width_keeps_fail_and_retains_raw_measurement():
    result = build_reading_core(5., 5.4, -.4, [], {"pump-8": 2.5}, .25)
    assert result.effective_width_m == -.4
    assert result.verdict == {"pump-8": "FAIL"}
    assert result.quality_flags == ["negative_effective_width"]


def test_quality_flags_do_not_relax_fail():
    result = build_reading_core(5., 4., 1., [], {"pump-8": 2.5}, .25,
                                quality_flags=["outside_calibration_depth"])
    assert result.verdict == {"pump-8": "FAIL"}


@pytest.mark.parametrize("flags", [[], ["outside_calibration_depth"]])
def test_export_refreshes_quality_metadata_without_mutating_input(flags):
    core = build_reading_core(5., 1., 4., [], {"pump-8": 2.5}, .25, quality_flags=flags)
    meta = {"adapter": "file", "measurement_quality": {"flags": ["stale"], "pass_blocked": True}}
    result = build_reading("quality_test", "edge", "frame.png", core, meta)
    assert result["source_meta"]["measurement_quality"] == {"flags": flags, "pass_blocked": bool(flags)}
    assert result["source_meta"]["adapter"] == "file"
    assert meta["measurement_quality"]["flags"] == ["stale"]


def test_extrapolated_motion_scale_cannot_make_empty_road_pass(monkeypatch):
    before = make_detection("승용차", 1., 10, 10, 91, 180)
    after = make_detection("승용차", 1., 40, 10, 91, 180)
    frames = iter([[before], [after]])
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: next(frames))
    frame = np.zeros((200, 300, 3), np.uint8)
    result = motion_demo.run_motion_aware_demo([frame, frame], 8., None, 200.)
    assert result.obstacle_width_m == 0.
    assert set(result.verdict.values()) == {"UNCERTAIN"}
    assert result.quality_flags == ["motion_scale_outside_calibration_depth"]
