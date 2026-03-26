"""
阶段 3 ─ 手部遮挡过滤（纯分析，不复制文件）

使用 MediaPipe Pose 提取关键点，判断手部是否遮挡面部/嘴部。
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════
#  手部遮挡检测器
# ═══════════════════════════════════════════════════════════════
class HandOcclusionDetector:
    """
    检测手部是否遮挡嘴部区域。
    使用像素级距离计算，避免宽高比畸变。
    """

    def __init__(self):
        self.backend = "mediapipe"
        self._pose = None
        self._mp_pose = None
        self._init()

    def _init(self):
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        logger.info("[手部遮挡] 后端: MediaPipe Pose")

    def is_occluded(self, frame) -> bool:
        """True = 手遮挡嘴部"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return False

        h, w = frame.shape[:2]
        lm = res.pose_landmarks.landmark

        # ── 嘴部位置（比鼻子更准确） ──
        mouth_left = lm[self._mp_pose.PoseLandmark.MOUTH_LEFT]
        mouth_right = lm[self._mp_pose.PoseLandmark.MOUTH_RIGHT]
        mouth_x = (mouth_left.x + mouth_right.x) / 2 * w
        mouth_y = (mouth_left.y + mouth_right.y) / 2 * h

        # ── 检测关键点：手腕 + 指尖 ──
        check_points = [
            self._mp_pose.PoseLandmark.LEFT_WRIST,
            self._mp_pose.PoseLandmark.RIGHT_WRIST,
            self._mp_pose.PoseLandmark.LEFT_INDEX,
            self._mp_pose.PoseLandmark.RIGHT_INDEX,
            self._mp_pose.PoseLandmark.LEFT_PINKY,
            self._mp_pose.PoseLandmark.RIGHT_PINKY,
        ]

        # 阈值：高度的 15%（像素级，与宽高比无关）
        threshold_px = 0.15 * h

        for pt_id in check_points:
            pt = lm[pt_id]
            if pt.visibility > 0.5:
                dx = pt.x * w - mouth_x
                dy = pt.y * h - mouth_y
                dist = np.hypot(dx, dy)
                if dist < threshold_px:
                    return True

        return False


# ── 全局单例 ──
_ho_det = None


def _get_ho_det():
    global _ho_det
    if _ho_det is None:
        _ho_det = HandOcclusionDetector()
    return _ho_det


# ═══════════════════════════════════════════════════════════════
#  对外接口（纯分析版）
# ═══════════════════════════════════════════════════════════════
def filter_hand_occlusion_analysis(
    video_path,
    max_occlusion_ratio=0.05,
    sample_interval=3,
):
    """
    纯分析：遍历视频帧，判断手部遮挡是否超标。

    Returns
    -------
    bool : True = 遮挡超标，应丢弃；False = 通过
    """
    det = _get_ho_det()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True  # 打不开 → 丢弃

    sampled = 0
    occluded = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % sample_interval != 0:
            continue

        sampled += 1
        if det.is_occluded(frame):
            occluded += 1

        # 早停
        if sampled > 10:
            cur_ratio = occluded / sampled
            if cur_ratio > max_occlusion_ratio * 3:
                cap.release()
                logger.info(f"  手部遮挡率 {cur_ratio:.0%} 超标，提前淘汰。")
                return True

    cap.release()

    if sampled == 0:
        return True  # 无帧 → 丢弃

    ratio = occluded / sampled
    if ratio <= max_occlusion_ratio:
        logger.info(f"  手部遮挡检查通过 ({ratio:.0%} ≤ {max_occlusion_ratio:.0%})")
        return False  # 通过
    else:
        logger.info(f"  手部遮挡率 {ratio:.0%} > {max_occlusion_ratio:.0%}，丢弃。")
        return True  # 丢弃
