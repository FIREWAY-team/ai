from copy import deepcopy

import numpy as np
import pytest

from scripts import restore_moran_widths as restore
from src.postprocess.passable_prob import build_reading_core


def fixtures(adapter="file"):
    cameras = {
        "cctv_2": {"still_url": "source.png", "wall_width_m": 5.6, "wall_width_source": "naver_map"},
        "cctv_moran_a18": {
            "still_url": "source.png", "wall_width_m": 6.08,
            "wall_width_source": "estimated_avg_of_similar_alleys",
        },
    }
    readings = [{
        "cctv_id": "cctv_moran_a18", "edge_id": "demo-edge", "still_public_url": "source.png",
        "wall_width_m": 6.08, "verdict": {"pump-8": "PASS"},
        "source_meta": {"adapter": adapter, "lat": 37.430393, "lon": 127.126909},
    }]
    return cameras, readings


def test_restore_rejudges_image_with_source_width_preserving_placement(monkeypatch):
    cameras, readings = fixtures()
    original = deepcopy((cameras, readings))

    def read_file(path, cctv_id, wall_width_m, edge_id):
        assert (path, cctv_id, edge_id) == ("source.png", "cctv_moran_a18", "demo-edge")
        core = build_reading_core(wall_width_m, 3.1, wall_width_m - 3.1, [], {"pump-8": 2.5}, .25)
        return restore.build_reading(cctv_id, edge_id, path, core)

    monkeypatch.setattr(restore, "read_from_file", read_file)
    updated_cameras, updated, changed = restore.restore_widths(cameras, readings)

    assert (cameras, readings) == original
    assert changed == ["cctv_moran_a18"]
    assert updated_cameras[changed[0]]["wall_width_m"] == 5.6
    assert updated[0]["verdict"] == {"pump-8": "UNCERTAIN"}
    assert updated[0]["effective_width_m"] == pytest.approx(2.5)
    assert updated[0]["source_meta"]["lat"] == readings[0]["source_meta"]["lat"]
    assert updated[0]["source_meta"]["width_reference_cctv_id"] == "cctv_2"


def test_restore_video_keeps_original_sampling_and_calibration_id(monkeypatch):
    cameras, readings = fixtures("video")
    frames = [np.zeros((100, 200, 3), dtype=np.uint8)] * 10

    def extract(path, max_frames, frame_interval_sec):
        assert (path, max_frames, frame_interval_sec) == ("source.png", 10, 1.0)
        return frames

    def judge(actual_frames, **kwargs):
        assert actual_frames is frames
        assert kwargs == {"wall_width_m": 5.6, "target_y_px": None,
                          "camera_height_px": 100., "cctv_id": "cctv_moran_a18"}
        return build_reading_core(5.6, 3.1, 2.5, [], {"pump-8": 2.5}, .25)

    monkeypatch.setattr(restore, "extract_frames", extract)
    monkeypatch.setattr(restore, "run_motion_aware_demo", judge)
    _, updated, _ = restore.restore_widths(cameras, readings)
    assert updated[0]["source_meta"]["adapter"] == "video"
    assert updated[0]["verdict"] == {"pump-8": "UNCERTAIN"}


def test_restore_does_not_promote_estimated_width_to_measured():
    cameras, readings = fixtures()
    cameras["cctv_2"]["wall_width_source"] = "estimated_avg_of_similar_alleys"
    assert restore.restore_widths(cameras, readings) == (cameras, readings, [])


@pytest.mark.parametrize("mismatch", ["reading_source", "conflicting_width"])
def test_restore_rejects_ambiguous_source_before_inference(monkeypatch, mismatch):
    cameras, readings = fixtures()
    if mismatch == "reading_source":
        readings[0]["still_public_url"] = "different.png"
    else:
        cameras["cctv_other"] = {**cameras["cctv_2"], "wall_width_m": 9.0}

    def fail(*args, **kwargs):
        pytest.fail("원본 검증 전에 추론 호출")

    monkeypatch.setattr(restore, "read_from_file", fail)
    with pytest.raises(ValueError):
        restore.restore_widths(cameras, readings)
