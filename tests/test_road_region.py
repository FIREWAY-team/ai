import json

import numpy as np
import pytest
import yaml

from scripts import accumulate_calibration, motion_demo
from src import camera_registry, pipeline
from src.inference import calibration_store, road_region, yolo
from src.inference.homography import local_scale_estimates, scale_estimates_with_history
from src.postprocess.passable_prob import compute_widths, find_narrowest_widths
from tests.helpers import make_detection


@pytest.fixture
def region_config(monkeypatch, tmp_path):
    path = tmp_path / "regions.yaml"
    config = {"regions": {"road": {
        "media_filename": "road.png", "frame_size": [300, 200],
        "polygon": [[0, 0], [149, 0], [149, 199], [0, 199]],
    }}}
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(road_region, "DEFAULT_REGIONS_PATH", path)
    cameras = {
        "road": {"still_url": "road.png"},
        "alias": {"still_url": "road.png", "calibration_source_cctv_id": "road"},
    }
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: cameras)
    return path, config, cameras


@pytest.fixture
def detections():
    return [
        make_detection("승용차", .9, 50, 50, 40, 100),
        make_detection("대형버스", .99, 200, 10, 90, 180),
    ]


def test_off_road_object_not_used_for_scale_or_width(region_config, detections, monkeypatch):
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: [])
    assert scale_estimates_with_history(detections, 200, "alias") == local_scale_estimates(detections[:1])
    expected = compute_widths(detections[:1], 100, 200, 5)
    assert compute_widths(detections, 100, 200, 5, "alias") == pytest.approx(expected)
    assert find_narrowest_widths(detections, 200, 5, "alias") == pytest.approx(
        find_narrowest_widths(detections[:1], 200, 5)
    )


def test_one_pixel_overlap_keeps_full_vehicle_mask(region_config):
    det = make_detection("승용차", .9, 149, 50, 50, 100)
    original = det.mask.copy()
    kept = road_region.filter_road_detections([det], "alias")
    assert len(kept) == 1 and kept[0] is det
    np.testing.assert_array_equal(kept[0].mask, original)


def test_no_region_preserves_existing_behavior(region_config, detections):
    assert len(road_region.filter_road_detections(detections, "unconfigured")) == 2
    assert len(road_region.filter_road_detections(detections, None)) == 2


def test_shape_mismatch_rejected_even_without_detections(region_config):
    with pytest.raises(ValueError, match="해상도"):
        road_region.filter_road_detections([], "alias", (100, 150, 3))
    with pytest.raises(ValueError, match="해상도"):
        road_region.filter_road_detections([make_detection("승용차", .9, 0, 0, 20, 40, (100, 150))], "alias")


@pytest.mark.parametrize("polygon", [[], [[0, 0], [300, 0], [0, 100]], [[0, 0], [0, 1], [0, 2]]])
def test_invalid_polygon_rejected(region_config, polygon):
    path, config, _ = region_config
    config["regions"]["road"]["polygon"] = polygon
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="도로 영역"):
        road_region.get_road_region("alias")


def test_wrong_source_rejected(region_config):
    _, _, cameras = region_config
    cameras["road"]["still_url"] = "different.png"
    with pytest.raises(ValueError, match="원본 영상"):
        road_region.get_road_region("road")


def test_history_isolated_by_polygon_version_and_shared_with_alias(region_config, tmp_path):
    path, config, _ = region_config
    legacy = tmp_path / calibration_store.SCALE_VERSION / "road.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"invalid": "must not read legacy history"}) + "\n")
    assert calibration_store.load_observations("alias", calibration_dir=tmp_path) == []
    observation = calibration_store.make_observation(.01, .9, 120, 200)
    calibration_store.append_observations("road", [observation], calibration_dir=tmp_path)
    assert calibration_store.load_observations("alias", calibration_dir=tmp_path) == [observation]
    first_version = road_region.get_road_region("road").version
    config["regions"]["road"]["polygon"][1][0] = 148
    path.write_text(yaml.safe_dump(config))
    assert road_region.get_road_region("road").version != first_version
    assert calibration_store.load_observations("alias", calibration_dir=tmp_path) == []
    assert legacy.exists()


def test_accumulation_uses_only_road_references(region_config, detections, monkeypatch, tmp_path):
    monkeypatch.setattr(accumulate_calibration, "_load_frames_for_still_url", lambda _: [np.zeros((200, 300, 3), np.uint8)])
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections)
    assert accumulate_calibration.accumulate_camera("alias", "road.png", tmp_path) == 1
    observations = calibration_store.load_observations("road", calibration_dir=tmp_path)
    assert observations[0].scale == pytest.approx(local_scale_estimates(detections[:1])[0][0])
    with pytest.raises(ValueError, match="원본이 아닌"):
        accumulate_calibration.accumulate_camera("road", "other.png", tmp_path)


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("target", [None, 100])
def test_image_and_video_keep_raw_detections_but_filter_width(region_config, detections, monkeypatch, video, target):
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections)
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: [])
    frame = np.zeros((200, 300, 3), np.uint8)
    if video:
        result = motion_demo.run_motion_aware_demo([frame, frame], 5, target, 200, cctv_id="alias")
    else:
        result = pipeline.process_frame(frame, 5, target, 200, cctv_id="alias")
    expected = compute_widths(detections[:1], 100, 200, 5)
    assert result.obstacle_width_m == pytest.approx(expected[1])
    assert len(result.detected_objects) == 2


@pytest.mark.parametrize("video", [False, True])
def test_all_excluded_does_not_become_pass_from_old_history(region_config, detections, monkeypatch, video):
    monkeypatch.setattr(yolo, "detect_vehicles", lambda _: detections[1:])
    monkeypatch.setattr(calibration_store, "load_observations", lambda _: [calibration_store.make_observation(.01, .9, 100, 200)])
    frame = np.zeros((200, 300, 3), np.uint8)
    with pytest.raises(ValueError, match="도로 영역"):
        if video:
            motion_demo.run_motion_aware_demo([frame, frame], 5, 100, 200, cctv_id="alias")
        else:
            pipeline.process_frame(frame, 5, 100, 200, cctv_id="alias")


@pytest.mark.parametrize("camera", ["cctv_3", "cctv_moran_a39"])
def test_a39_region_keeps_alley_vehicles_but_excludes_fenced_parking(monkeypatch, camera):
    media = "cctv_atypical_road_000481_00001.png"
    cameras = {
        "cctv_3": {"still_url": media},
        "cctv_moran_a39": {"still_url": media, "calibration_source_cctv_id": "cctv_3"},
    }
    monkeypatch.setattr(camera_registry, "load_cameras", lambda path: cameras)
    # 원본에서 확인한 세 위치. 울타리에 걸친 골목 차량은 마스크 전체 유지.
    left = make_detection("승용차", .9, 289, 582, 360, 395, (1080, 1920))
    boundary = make_detection("승용차", .8, 1029, 441, 166, 162, (1080, 1920))
    parking = make_detection("승용차", .9, 1365, 541, 408, 221, (1080, 1920))
    original = boundary.mask.copy()
    kept = road_region.filter_road_detections([left, parking, boundary], camera, (1080, 1920, 3))
    assert len(kept) == 2 and kept[0] is left and kept[1] is boundary
    np.testing.assert_array_equal(kept[1].mask, original)
