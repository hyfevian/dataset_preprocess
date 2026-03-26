"""
阶段 2 ─ 跳切检测 / 人脸位置分析
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════
#  1. 跳切检测
# ═══════════════════════════════════════════════════════════════
def _color_hist(frame, bins=64):
    """HSV 颜色直方图"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detect_jump_cuts(
    video_path,
    flow_sudden_threshold=25.0,
    hist_threshold=0.35,
    min_consecutive=2,
    sample_interval=2,
    scale=0.25,
):
    """
    跳切检测 ── 使用光流突变 AND 颜色直方图突变（双重确认减少误报）。

    需要连续 min_consecutive 帧同时满足两个条件才判定为跳切。

    Returns
    -------
    bool : True = 检测到跳切，应丢弃
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True

    ret, frame1 = cap.read()
    if not ret:
        cap.release()
        return True

    small = cv2.resize(frame1, None, fx=scale, fy=scale)
    prvs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    prev_hist = _color_hist(small)
    prev_mag = 0.0
    consec = 0
    idx = 0

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        idx += 1
        if idx % sample_interval != 0:
            continue

        small2 = cv2.resize(frame2, None, fx=scale, fy=scale)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

        # ── 光流 ──
        flow = cv2.calcOpticalFlowFarneback(
            prvs, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        mag_change = abs(mean_mag - prev_mag)

        # ── 直方图 ──
        cur_hist = _color_hist(small2)
        hist_diff = cv2.compareHist(prev_hist, cur_hist,
                                    cv2.HISTCMP_BHATTACHARYYA)

        # ── AND 逻辑：两个信号同时异常才计数 ──
        anomaly = (mag_change > flow_sudden_threshold) and \
                  (hist_diff > hist_threshold)

        if anomaly:
            consec += 1
            if consec >= min_consecutive:
                cap.release()
                return True
        else:
            consec = 0

        prev_mag = mean_mag
        prev_hist = cur_hist
        prvs = gray2

    cap.release()
    return False


# ═══════════════════════════════════════════════════════════════
#  2. 人脸检测器（MediaPipe）
# ═══════════════════════════════════════════════════════════════
class _FaceDetector:
    """强制使用 MediaPipe 人脸检测"""

    def __init__(self):
        self.backend = None
        self._init()

    def _init(self):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "mediapipe 版本不包含 solutions API。"
                "请安装: pip install mediapipe==0.10.14"
            )
        self._mp_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.backend = "mediapipe"
        logger.info("[人脸检测] 后端: MediaPipe")

    def detect(self, frame):
        """返回 list[(x, y, w, h)]"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._mp_det.process(rgb)
        if not res.detections:
            return []
        fh, fw = frame.shape[:2]
        out = []
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x = int(bb.xmin * fw)
            y = int(bb.ymin * fh)
            w = int(bb.width * fw)
            h = int(bb.height * fh)
            out.append((x, y, w, h))
        return out


# 全局单例
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = _FaceDetector()
    return _detector


# ═══════════════════════════════════════════════════════════════
#  3. 人脸位置分析（纯分析，不编码）
# ═══════════════════════════════════════════════════════════════
def analyze_face_positions(video_path, target_size=512,
                           max_no_face_ratio=0.15,
                           sample_interval=3):
    """
    扫描视频帧，收集人脸位置，计算全局裁剪参数。

    Returns
    -------
    tuple (x, y, w, h) : 裁剪参数（原始帧坐标系）
    None : 人脸不合格，应丢弃
    """
    det = _get_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    centers_x, centers_y, sizes = [], [], []
    total = 0
    sampled = 0
    no_face = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        if total % sample_interval != 1:
            continue

        sampled += 1
        faces = det.detect(frame)
        if not faces:
            no_face += 1
            continue

        # 取最大人脸
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        centers_x.append(fx + fw // 2)
        centers_y.append(fy + fh // 2)
        sizes.append(max(fw, fh))

    cap.release()

    if sampled == 0 or not centers_x:
        logger.info(f"  无可用人脸帧: {video_path}")
        return None

    ratio = no_face / sampled
    if ratio > max_no_face_ratio:
        logger.info(f"  无脸帧比例 {ratio:.0%} > {max_no_face_ratio:.0%}，丢弃。")
        return None

    # ── 计算全局裁剪框 ──
    med_cx = int(np.median(centers_x))
    med_cy = int(np.median(centers_y))
    face_p90 = int(np.percentile(sizes, 90))

    # 读取帧尺寸
    cap2 = cv2.VideoCapture(video_path)
    ret, f0 = cap2.read()
    cap2.release()
    if not ret:
        return None
    frame_h, frame_w = f0.shape[:2]

    # 正方形裁剪，边长不超过最短边
    max_possible_crop = min(frame_w, frame_h)
    crop_size = max(target_size, int(face_p90 * 2.5))
    crop_size = min(crop_size, max_possible_crop)

    half = crop_size // 2
    # 限制中心点，使裁剪框完全在画面内
    cx = max(half, min(med_cx, frame_w - half))
    cy = max(half, min(med_cy, frame_h - half))

    x1 = cx - half
    y1 = cy - half

    # 边界安全校验
    x2 = min(frame_w, x1 + crop_size)
    y2 = min(frame_h, y1 + crop_size)
    x1 = max(0, x2 - crop_size)
    y1 = max(0, y2 - crop_size)

    cw = x2 - x1
    ch = y2 - y1

    logger.info(f"  人脸分析完成: center=({cx},{cy}), crop=({cw}x{ch}), "
                f"face_p90={face_p90}, 无脸率={ratio:.0%}")

    return (x1, y1, cw, ch)
