from copy import deepcopy

import pytest

from scripts.prepare_measurement_review import validate_submission


@pytest.fixture
def sample():
    camera = {
        "source_cctv_id": "cctv_test", "moran_cctv_id": "moran_test",
        "media_filename": "test.png", "media_sha256": "media-hash", "frame_sha256": "frame-hash",
        "frame_size": [600, 400], "sample_index": 0, "wall_width_m": 5.6,
    }
    annotation = {**deepcopy(camera), "points": [[20, 200], [500, 200]],
                  "matches_measured_segment": True, "status": "annotated"}
    return {"schema_version": 1, "annotations": [annotation]}, {"cameras": [camera]}


def test_annotation_preserves_full_road_width_and_pixel_endpoints(sample):
    payload, manifest = sample
    before = deepcopy(payload)
    result = validate_submission(payload, manifest)
    assert result[0]["wall_width_m"] == 5.6
    assert result[0]["points"] == [[20, 200], [500, 200]]
    assert payload == before


@pytest.mark.parametrize("key,value", [
    ("media_sha256", "other-media"), ("frame_sha256", "other-frame"),
    ("frame_size", [300, 200]), ("wall_width_m", 4.), ("sample_index", 9),
    ("moran_cctv_id", "other-camera"),
])
def test_wrong_source_frame_or_measured_width_rejected(sample, key, value):
    payload, manifest = sample
    payload["annotations"][0][key] = value
    with pytest.raises(ValueError, match="식별값"):
        validate_submission(payload, manifest)


@pytest.mark.parametrize("points", [
    [[-1, 200], [500, 200]], [[20, 400], [500, 200]],
    [[20, 200], [600, 200]], [[float("nan"), 200], [500, 200]],
    [[True, 200], [500, 200]], [[20, 200], [20, 200]],
])
def test_invalid_or_coincident_coordinates_rejected(sample, points):
    payload, manifest = sample
    payload["annotations"][0]["points"] = points
    with pytest.raises(ValueError):
        validate_submission(payload, manifest)


@pytest.mark.parametrize("status,points", [("pending", []), ("pending", [[20, 200]]), ("unavailable", [])])
def test_incomplete_annotations_can_be_saved_as_drafts(sample, status, points):
    payload, manifest = sample
    payload["annotations"][0].update(status=status, points=points, matches_measured_segment=False)
    assert validate_submission(payload, manifest)[0]["status"] == status


def test_annotation_requires_explicit_measured_segment_match(sample):
    payload, manifest = sample
    payload["annotations"][0]["matches_measured_segment"] = False
    with pytest.raises(ValueError, match="실측 구간"):
        validate_submission(payload, manifest)


def test_duplicate_camera_rejected(sample):
    payload, manifest = sample
    second = {**manifest["cameras"][0], "source_cctv_id": "second"}
    manifest["cameras"].append(second)
    payload["annotations"].append(deepcopy(payload["annotations"][0]))
    with pytest.raises(ValueError, match="중복"):
        validate_submission(payload, manifest)
