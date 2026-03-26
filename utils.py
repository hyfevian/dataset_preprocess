"""
公共工具函数
"""

import os
import json
import subprocess
import logging


# ═══════════════════════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════════════════════
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger("utils", "preprocess.log")


# ═══════════════════════════════════════════════════════════════
#  文件操作
# ═══════════════════════════════════════════════════════════════
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".ts"}


def cleanup(path):
    """安全删除文件"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════
#  视频校验 & 信息获取（统一使用 ffprobe）
# ═══════════════════════════════════════════════════════════════
def validate_video(video_path):
    """用 ffprobe 验证视频文件是否可正常解码"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffprobe_path = os.path.join(base_dir, "ffprobe.exe")
    cmd = [
        ffprobe_path if os.path.exists(ffprobe_path) else "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False
        info = json.loads(r.stdout)
        streams = info.get("streams", [])
        if not streams:
            return False
        return streams[0].get("width", 0) > 0
    except Exception:
        return False


def get_video_info_ffprobe(video_path):
    """
    使用 ffprobe 获取视频信息。
    返回 (fps, total_frames, duration)，失败返回 (None, None, None)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffprobe_path = os.path.join(base_dir, "ffprobe.exe")
    cmd = [
        ffprobe_path if os.path.exists(ffprobe_path) else "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None, None, None

        info = json.loads(r.stdout)
        streams = info.get("streams", [])
        if not streams:
            return None, None, None

        stream = streams[0]

        # 解析 fps（分数形式 "25/1" 或 "30000/1001"）
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, r_frame_rate.split("/"))
            fps = num / den if den != 0 else 0
        except (ValueError, ZeroDivisionError):
            fps = 0

        if fps <= 0:
            return None, None, None

        # 解析时长
        duration = None
        if "duration" in stream and stream["duration"] != "N/A":
            try:
                duration = float(stream["duration"])
            except (ValueError, TypeError):
                pass

        if duration is None:
            fmt = info.get("format", {})
            if "duration" in fmt:
                try:
                    duration = float(fmt["duration"])
                except (ValueError, TypeError):
                    pass

        if duration is None or duration <= 0:
            return fps, None, None

        # 解析帧数
        nb_frames = None
        if "nb_frames" in stream and stream["nb_frames"] != "N/A":
            try:
                nb_frames = int(stream["nb_frames"])
            except (ValueError, TypeError):
                pass

        if nb_frames is None or nb_frames <= 0:
            nb_frames = int(fps * duration)

        return fps, nb_frames, duration

    except Exception as e:
        logger.warning(f"ffprobe 获取信息失败: {video_path} ({e})")
        return None, None, None


# ═══════════════════════════════════════════════════════════════
#  单次编码（FPS + crop + scale + 音频标准化）
# ═══════════════════════════════════════════════════════════════
def encode_single_pass(input_path, output_path,
                       crop_params=None,
                       target_fps=25,
                       target_size=512,
                       crf=18,
                       audio_rate=16000):
    """
    一次 ffmpeg 调用完成：FPS 转换 + 裁剪 + 缩放 + 音频标准化。

    Parameters
    ----------
    crop_params : tuple (x, y, w, h) or None
        裁剪参数，None 则不裁剪。
    target_fps : int
        目标帧率。
    target_size : int
        输出正方形边长。
    crf : int
        编码质量。
    audio_rate : int
        音频采样率（wav2vec 需要 16000）。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 构建 video filter chain
    vf_parts = []
    vf_parts.append(f"fps={target_fps}")

    if crop_params is not None:
        cx, cy, cw, ch = crop_params
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")

    vf_parts.append(f"scale={target_size}:{target_size}:flags=lanczos")
    vf_str = ",".join(vf_parts)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")

    cmd = [
        ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", input_path,
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k", "-ar", str(audio_rate), "-ac", "1",
        "-max_muxing_queue_size", "1024",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=180,
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"FFmpeg 编码超时: {input_path}")
        cleanup(output_path)
        return False
    except subprocess.CalledProcessError:
        pass  # 进入降级

    # ── 降级方案 ──
    logger.info(f"  常规编码失败，尝试降级...")
    cmd_fallback = [
        ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts+discardcorrupt",
        "-i", input_path,
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k", "-ar", str(audio_rate), "-ac", "1",
        "-max_muxing_queue_size", "2048",
        "-threads", "1",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(
            cmd_fallback, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300,
        )
        logger.info(f"  降级方案成功: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"  降级方案也失败 ({e})，放弃此视频。")
        cleanup(output_path)
        return False
