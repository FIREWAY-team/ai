import numpy as np
import pytest

from scripts import motion_demo, refresh_moran_readings
from src import camera_registry, pipeline
from src.inference import calibration_store, yolo
from src.inference.yolo import VehicleDetection
from tests.helpers import make_detection, make_rotated_rect_mask


@pytest.fixture
def width_scene(monkeypatch):
    cameras = {"width_test": {
        "wall_width_m": 6.08, "wall_width_source": "estimated_avg_of_similar_alleys",
        "still_url": "scene.png",
    }}
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: cameras)
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: [])
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: [make_detection("승용차", .9, 10, 10, 91, 180)])
    return cameras, np.zeros((200, 300, 3), np.uint8)


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("target", [None, 189.])
@pytest.mark.parametrize("source, expected", [
    ("estimated_avg_of_similar_alleys", "UNCERTAIN"),
    (None, "UNCERTAIN"), ("unrecognized_source", "UNCERTAIN"),
    ("naver_map", "PASS"), ("kakao_map", "PASS"), ("field_measurement", "PASS"),
])
def test_only_measured_registered_width_can_support_pass(width_scene, video, target, source, expected):
    cameras, frame = width_scene
    cameras["width_test"]["wall_width_source"] = source
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 6.08, target, 200., cctv_id="width_test")
    else:
        result = pipeline.process_frame(frame, 6.08, target, 200., cctv_id="width_test")
    assert result.effective_width_m == pytest.approx(4.28)
    assert set(result.verdict.values()) == {expected}
    assert result.quality_flags == (["unverified_wall_width"] if expected == "UNCERTAIN" else [])


@pytest.mark.parametrize("video", [False, True])
def test_unverified_width_does_not_relax_existing_fail(width_scene, video):
    cameras, frame = width_scene
    cameras["width_test"]["wall_width_m"] = 3.8
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 3.8, None, 200., cctv_id="width_test")
    else:
        result = pipeline.process_frame(frame, 3.8, None, 200., cctv_id="width_test")
    assert set(result.verdict.values()) == {"FAIL"}
    assert result.quality_flags == ["unverified_wall_width"]


def test_measured_label_cannot_hide_different_input_width_or_missing_camera(width_scene):
    cameras, _ = width_scene
    cameras["width_test"]["wall_width_source"] = "naver_map"
    assert camera_registry.wall_width_quality_flags("width_test", 8.) == ["unverified_wall_width"]
    assert camera_registry.wall_width_quality_flags("missing", 6.08) == ["unverified_wall_width"]


def test_alias_cannot_claim_measurement_when_original_is_estimated(width_scene):
    cameras, _ = width_scene
    cameras["alias"] = {**cameras["width_test"], "wall_width_source": "naver_map",
                        "calibration_source_cctv_id": "width_test"}
    assert camera_registry.wall_width_quality_flags("alias", 6.08) == ["unverified_wall_width"]
    cameras["width_test"]["wall_width_source"] = "naver_map"
    assert camera_registry.wall_width_quality_flags("alias", 6.08) == []
    cameras["width_test"]["wall_width_m"] = 7.
    assert camera_registry.wall_width_quality_flags("alias", 6.08) == ["unverified_wall_width"]


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("target", [None, 189.])
@pytest.mark.parametrize("width, expected", [(6.08, "PASS"), (3.8, "FAIL")])
def test_demo_uses_calculated_verdict_with_explicit_estimated_width(width_scene, video, target, width, expected):
    cameras, frame = width_scene
    cameras["width_test"]["wall_width_m"] = width
    if video:
        result = motion_demo.run_motion_aware_demo(
            [frame, frame], width, target, 200., cctv_id="width_test", allow_estimated_wall_width=True,
        )
    else:
        result = pipeline.process_frame(
            frame, width, target, 200., cctv_id="width_test", allow_estimated_wall_width=True,
        )
    assert set(result.verdict.values()) == {expected}
    assert result.effective_width_m == pytest.approx(width - 1.8)


@pytest.mark.parametrize("source, input_width", [(None, 6.08), ("unknown", 6.08), ("estimated_avg_of_similar_alleys", 9.)])
def test_demo_does_not_accept_unknown_source_or_mismatched_width(width_scene, source, input_width):
    cameras, _ = width_scene
    cameras["width_test"]["wall_width_source"] = source
    assert camera_registry.wall_width_quality_flags(
        "width_test", input_width, allow_estimated_wall_width=True,
    ) == ["unverified_wall_width"]


@pytest.mark.parametrize("video", [False, True])
def test_demo_keeps_other_measurement_quality_restrictions(width_scene, monkeypatch, video):
    cameras, _ = width_scene
    cameras["width_test"]["wall_width_m"] = 8.
    detections = [
        make_detection("승용차", .9, 400, 10, 91, 100, (500, 600)),
        VehicleDetection("승용차", .9, (70, 250, 180, 350),
                         make_rotated_rect_mask((500, 600), (120, 300), 40, 90, 45)),
    ]
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections)
    frame = np.zeros((500, 600, 3), np.uint8)
    if video:
        result = motion_demo.run_motion_aware_demo(
            [frame, frame], 8., None, 500., cctv_id="width_test", allow_estimated_wall_width=True,
        )
    else:
        result = pipeline.process_frame(
            frame, 8., None, 500., cctv_id="width_test", allow_estimated_wall_width=True,
        )
    assert result.effective_width_m > 5.
    assert result.quality_flags == ["outside_calibration_depth"]
    assert set(result.verdict.values()) == {"UNCERTAIN"}


@pytest.mark.parametrize("allow_estimated", [False, True])
def test_refresh_replaces_stale_measurement_claim_and_exposes_actual_width_source(width_scene, monkeypatch, allow_estimated):
    _, frame = width_scene
    monkeypatch.setattr(refresh_moran_readings, "load_frame", lambda _: frame)
    original = {"cctv_id": "width_test", "edge_id": "edge", "still_public_url": "scene.png",
                "source_meta": {"adapter": "file", "wall_width_source": "naver_map",
                                "footage_note": "실측 검증된 유사 골목 영상"}}
    result = refresh_moran_readings.refresh_readings([original], allow_estimated_wall_width=allow_estimated)[0]
    assert result["source_meta"]["wall_width_source"] == "estimated_avg_of_similar_alleys"
    assert "실측 검증된" not in result["source_meta"]["footage_note"]
    assert result["source_meta"]["measurement_quality"] == {
        "flags": [] if allow_estimated else ["unverified_wall_width"], "pass_blocked": not allow_estimated,
    }
    assert result["source_meta"]["decision_policy"]["allow_estimated_wall_width"] is allow_estimated
    assert set(result["verdict"].values()) == ({"PASS"} if allow_estimated else {"UNCERTAIN"})
    assert original["source_meta"]["wall_width_source"] == "naver_map"
