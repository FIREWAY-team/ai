from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.inference import yolo


def prediction(masks, boxes):
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: masks))
    return SimpleNamespace(masks=SimpleNamespace(data=tensor), boxes=boxes, names={0: "승용차"})


def vehicle_box():
    return SimpleNamespace(
        cls=np.array([0]), conf=np.array([.8]), xyxy=np.array([[20., 10., 60., 40.]])
    )


@pytest.mark.parametrize("shape", [(108, 192), (192, 108), (128, 128)])
def test_detection_requests_unpadded_masks_and_preserves_original_pixels(monkeypatch, shape):
    frame = np.zeros((*shape, 3), np.uint8)
    masks = np.zeros((1, *shape), np.float32)
    masks[0, 10:40, 20:60] = 1
    model = Mock()
    model.predict.return_value = [prediction(masks, [vehicle_box()])]
    monkeypatch.setattr(yolo, "_load_model", lambda _: model)

    detections = yolo.detect_vehicles(frame)

    assert model.predict.call_args.kwargs["retina_masks"] is True
    assert len(detections) == 1
    assert detections[0].mask.dtype == bool
    np.testing.assert_array_equal(detections[0].mask, masks[0].astype(bool))
    assert detections[0].bbox == (20., 10., 60., 40.)
    assert detections[0].confidence == pytest.approx(.8)
    assert detections[0].vehicle_class == "승용차"


@pytest.mark.parametrize("mask_shape", [(1, 64, 64), (2, 108, 192), (108, 192)])
def test_inconsistent_mask_geometry_stops_measurement(monkeypatch, mask_shape):
    model = Mock()
    model.predict.return_value = [prediction(np.zeros(mask_shape), [vehicle_box()])]
    monkeypatch.setattr(yolo, "_load_model", lambda _: model)
    with pytest.raises(ValueError, match="원본 좌표 마스크"):
        yolo.detect_vehicles(np.zeros((108, 192, 3), np.uint8))


def test_empty_scene_needs_no_mask_conversion(monkeypatch):
    model = Mock()
    model.predict.return_value = [SimpleNamespace(masks=None)]
    monkeypatch.setattr(yolo, "_load_model", lambda _: model)
    assert yolo.detect_vehicles(np.zeros((108, 192, 3), np.uint8)) == []
