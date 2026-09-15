from __future__ import annotations

import cv2
import numpy as np

from src.adapters.video_adapter import extract_frames


def _write_test_video(path, n_frames: int, fps: float, size: tuple[int, int] = (64, 48)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_picks_frames_at_interval(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path, n_frames=30, fps=10.0)  # 3초 분량, 10fps

    frames = extract_frames(str(video_path), max_frames=3, frame_interval_sec=1.0)
    # 10fps * 1초 간격 = 10프레임마다 하나 -> 0, 10, 20번째 프레임 총 3장
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (48, 64, 3)


def test_extract_frames_respects_max_frames_cap(tmp_path):
    video_path = tmp_path / "test2.mp4"
    _write_test_video(video_path, n_frames=100, fps=5.0)

    frames = extract_frames(str(video_path), max_frames=4, frame_interval_sec=0.2)
    assert len(frames) == 4


def test_extract_frames_raises_for_missing_file(tmp_path):
    try:
        extract_frames(str(tmp_path / "missing.mp4"))
    except ValueError:
        pass
    else:
        raise AssertionError("존재하지 않는 파일이면 에러여야 합니다")


def test_extract_frames_raises_when_video_has_no_readable_frames(tmp_path):
    # 정상적으로 열리지만(헤더는 있음) 프레임이 하나도 없는 케이스를
    # 빈 파일로 흉내낸다 — cv2가 못 열어서 어차피 ValueError로 귀결되지만,
    # "빈 동영상"과 "존재하지 않는 파일" 둘 다 같은 방식으로 실패해야 함을 확인.
    empty_path = tmp_path / "empty.mp4"
    empty_path.write_bytes(b"")
    try:
        extract_frames(str(empty_path))
    except ValueError:
        pass
    else:
        raise AssertionError("빈 동영상 파일이면 에러여야 합니다")
