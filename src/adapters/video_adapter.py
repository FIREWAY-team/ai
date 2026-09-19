"""입력 어댑터 — 로컬 동영상 파일 (스펙 0장). 저장된 mp4 등에서 연속 프레임
여러 장을 뽑아 모션 인식 판독(scripts/motion_demo.run_motion_aware_demo)에
넘긴다. RTSP 어댑터가 "실시간 스트림에서 한 장"을 뽑는 것과 짝을 이루는
구조 — 이쪽은 "저장된 동영상에서 여러 장"을 뽑는다.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MAX_FRAMES = 10
DEFAULT_FRAME_INTERVAL_SEC = 1.0
FALLBACK_FPS = 5.0  # 이 프로젝트가 다루는 CCTV 데이터셋의 실제 녹화 fps(CSV CCTV_TYPE 필드에서 확인)


def extract_frames(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_interval_sec: float = DEFAULT_FRAME_INTERVAL_SEC,
) -> list[np.ndarray]:
    """동영상 파일에서 frame_interval_sec 간격으로 최대 max_frames장을 뽑는다.

    run_motion_aware_demo()가 "같은 시간 간격으로 떨어진 프레임"을 전제로
    속도(m/s)를 계산하므로(classify_motion), 이 함수가 실제 fps를 읽어서
    그 간격에 맞는 스텝으로 골라내는 역할을 한다 — 매 프레임을 다 넘기면
    안 되고(움직임 판별이 프레임 간격에 의존), 그렇다고 임의로 아무 프레임이나
    골라도 안 된다.

    fps를 못 읽으면(컨테이너에 메타데이터가 없는 경우) FALLBACK_FPS로
    근사한다 — 실제 간격이 다르면 속도 추정이 그만큼 부정확해지므로,
    가능하면 호출부가 실제 fps를 알고 있을 때 frame_interval_sec를
    직접 맞춰 넘기는 걸 권장한다.
    """
    import cv2  # 무거운 의존성 — 실제 사용 시점에만 로드(다른 어댑터와 동일한 관례)

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = FALLBACK_FPS
        frame_step = max(1, round(fps * frame_interval_sec))

        frames: list[np.ndarray] = []
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % frame_step == 0:
                frames.append(frame)
            idx += 1

        if not frames:
            raise ValueError(f"동영상에서 프레임을 읽지 못했습니다: {video_path}")
        return frames
    finally:
        cap.release()
