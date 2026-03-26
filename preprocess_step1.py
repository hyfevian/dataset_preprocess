"""
阶段 1 ─ 去重 / 场景切片
"""

import os
import hashlib
import glob
import subprocess
import json
import logging

from utils import VIDEO_EXTS, validate_video, get_video_info_ffprobe, cleanup

logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════
#  1. MD5 去重
# ═══════════════════════════════════════════════════════════════
def calculate_md5(file_path, chunk_size=65536):
    """计算文件 MD5，使用 64KB 块提升 IO 效率"""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"读取文件失败: {file_path} ({e})")
        return None


def deduplicate_videos(input_dir, output_dir):
    """
    递归扫描 input_dir 下所有视频文件，通过 MD5 去重。
    返回去重后的原始路径列表（不复制文件）。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── MD5 缓存 ──
    cache_file = os.path.join(output_dir, "md5_cache.json")
    md5_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                md5_cache = json.load(f)
        except Exception:
            md5_cache = {}

    # ── 递归收集视频 ──
    video_files = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                video_files.append(os.path.join(root, fn))
    video_files.sort()

    seen_hashes = set()
    unique = []
    cache_dirty = False

    for vf in video_files:
        try:
            st = os.stat(vf)
            cache_key = f"{os.path.abspath(vf)}|{st.st_size}|{st.st_mtime}"
        except OSError:
            continue

        if cache_key in md5_cache:
            fhash = md5_cache[cache_key]
        else:
            fhash = calculate_md5(vf)
            if fhash:
                md5_cache[cache_key] = fhash
                cache_dirty = True

        if fhash and fhash not in seen_hashes:
            seen_hashes.add(fhash)
            unique.append(vf)

    if cache_dirty:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(md5_cache, f)
        except Exception as e:
            logger.warning(f"保存 MD5 缓存失败: {e}")

    logger.info(f"扫描 {len(video_files)} 个文件，去重后 {len(unique)} 个")
    return unique


# ═══════════════════════════════════════════════════════════════
#  2. 场景切片
# ═══════════════════════════════════════════════════════════════
def _remux_copy(src, dst, timeout=60):
    """用 ffmpeg 重新封装（不重编码）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")
    cmd = [
        ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts",
        "-i", src,
        "-c", "copy",
        "-movflags", "+faststart",
        dst,
    ]
    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return True
    except Exception:
        cleanup(dst)
        return False


def _split_by_time(video_path, output_dir, base_name,
                   total_dur, max_dur, min_dur):
    """将超长视频按固定时长切分（stream copy，不重编码）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")
    segments = []
    idx = 1
    cur = 0.0
    while cur < total_dur:
        seg_dur = min(max_dur, total_dur - cur)
        if seg_dur < min_dur:
            break
        out = os.path.join(output_dir, f"{base_name}_scene_{idx:03d}.mp4")
        cmd = [
            ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-ss", f"{cur:.3f}",
            "-i", video_path,
            "-t", f"{seg_dur:.3f}",
            "-c", "copy",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            out,
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=180,
            )
            if validate_video(out):
                segments.append(out)
            else:
                # stream copy 失败，尝试重编码
                cmd_enc = [
                    ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y",
                    "-err_detect", "ignore_err",
                    "-ss", f"{cur:.3f}",
                    "-i", video_path,
                    "-t", f"{seg_dur:.3f}",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    out,
                ]
                subprocess.run(
                    cmd_enc, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=180,
                )
                if validate_video(out):
                    segments.append(out)
                else:
                    cleanup(out)
        except Exception:
            cleanup(out)
        cur += seg_dur
        idx += 1
    return segments


def slice_video(video_path, output_dir,
                min_duration=3.0, max_duration=60.0):
    """
    使用 PySceneDetect 将视频分割为连贯片段。
    尽量使用 stream copy（无损），只在必要时重编码。
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # ── 跳过已切片（精确匹配） ──
    existing = glob.glob(os.path.join(output_dir, f"{base_name}_scene_*.*"))
    if existing:
        valid = [f for f in existing if validate_video(f)]
        if valid:
            return valid
        # 存在但无效 → 清理后重做
        for f in existing:
            cleanup(f)

    # ── 文件校验 ──
    if not validate_video(video_path):
        logger.warning(f"视频文件损坏/不可读: {video_path}")
        return []

    # ── 场景检测 ──
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(video_path, ContentDetector(threshold=27.0))
    except Exception as e:
        logger.warning(f"场景检测出错: {e}")
        return []

    # ── 无场景切换 ──
    if not scene_list:
        fps, nframes, duration = get_video_info_ffprobe(video_path)
        if duration is None or duration <= 0:
            return []

        if min_duration <= duration <= max_duration:
            out = os.path.join(output_dir, f"{base_name}_scene_001.mp4")
            if _remux_copy(video_path, out):
                logger.info(f"  无场景切换({duration:.1f}s)，重封装保留。")
                return [out]
            else:
                logger.info("  重封装失败，尝试重编码...")
                base_dir = os.path.dirname(os.path.abspath(__file__))
                ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")
                cmd = [
                    ffmpeg_path if os.path.exists(ffmpeg_path) else "ffmpeg", "-y", "-err_detect", "ignore_err",
                    "-i", video_path,
                    "-c:v", "libx264", "-crf", "18",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", out,
                ]
                try:
                    subprocess.run(cmd, check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=180)
                    return [out]
                except Exception:
                    cleanup(out)
                    return []

        elif duration > max_duration:
            logger.info(f"  时长 {duration:.1f}s > {max_duration}s，按时长分段。")
            return _split_by_time(
                video_path, output_dir, base_name,
                duration, max_duration, min_duration,
            )
        else:
            logger.info(f"  时长 {duration:.1f}s < {min_duration}s，跳过。")
            return []

    # ── 有场景切换：筛选并切分 ──
    valid_scenes = [
        s for s in scene_list
        if min_duration <= s[1].get_seconds() - s[0].get_seconds() <= max_duration
    ]
    if not valid_scenes:
        logger.info(f"  没有符合时长要求 ({min_duration}~{max_duration}s) 的场景。")
        return []

    try:
        from scenedetect import split_video_ffmpeg
        tpl = os.path.join(output_dir, f"{base_name}_scene_$SCENE_NUMBER.mp4")
        split_video_ffmpeg(video_path, valid_scenes, tpl, show_progress=False)
    except Exception as e:
        logger.warning(f"场景切分失败: {e}")
        return []

    results = glob.glob(os.path.join(output_dir, f"{base_name}_scene_*.*"))
    # 验证所有切片
    valid_results = []
    for r in results:
        if validate_video(r):
            valid_results.append(r)
        else:
            cleanup(r)
    return valid_results