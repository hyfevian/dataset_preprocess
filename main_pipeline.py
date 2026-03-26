"""
主管道 ─ 视频数据集预处理
"""

import os
import sys
import json
import time
import argparse
import tempfile
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from preprocess_step1 import deduplicate_videos, slice_video
from preprocess_step2 import detect_jump_cuts, analyze_face_positions
from preprocess_step3 import filter_hand_occlusion_analysis
from preprocess_step4 import filter_syncnet
from utils import validate_video, cleanup, encode_single_pass, setup_logger, get_video_info_ffprobe

# ─────────────────────── 日志 ───────────────────────────
logger = setup_logger("pipeline", "preprocess.log")

# ─────────────────────── 常量 ───────────────────────────
STAGES = [
    "去重", "切片", "跳切检测",
    "人脸分析", "手部遮挡", "编码输出", "SyncNet"
]


# ═══════════════════════════════════════════════════════════════
#  状态管理（原子写入）
# ═══════════════════════════════════════════════════════════════
def load_status(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "videos": {},
        "stats": {
            "total_videos": 0,
            "total_clips": 0,
            "clips_ffmpeg_fail": 0,
            "clips_jumpcut_reject": 0,
            "clips_noface_reject": 0,
            "clips_hand_reject": 0,
            "clips_sync_reject": 0,
            "clips_passed": 0,
            "videos_with_valid_clips": 0,
            "videos_all_rejected": 0,
        },
    }


def save_status(path, data):
    """原子写入，防止断电损坏"""
    dir_name = os.path.dirname(path) or "."
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp", dir=dir_name,
            delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"保存状态文件失败: {e}")


# ═══════════════════════════════════════════════════════════════
#  单个 clip 处理（分析 → 单次编码 → SyncNet）
# ═══════════════════════════════════════════════════════════════
def process_single_clip(clip_path, output_dir, target_fps=25, target_size=512,
                        skip_jumpcut=False, skip_face=False, skip_hand=False, skip_syncnet=False):
    """
    返回 (final_path_or_None, reject_reason_or_"passed")

    Phase A: 纯分析（不写文件）
    Phase B: 单次 ffmpeg 编码
    Phase C: SyncNet 过滤
    """
    clip_name = os.path.basename(clip_path)

    # ── Phase A1: 跳切检测 ──
    if not skip_jumpcut:
        logger.info(f"  [{STAGES[2]}] {clip_name}")
        if detect_jump_cuts(clip_path):
            logger.info(f"  → 跳切淘汰: {clip_name}")
            return None, "jumpcut"
    else:
        logger.info(f"  [跳过 {STAGES[2]}] {clip_name}")

    # ── Phase A2: 人脸位置分析 ──
    crop_params = None
    if not skip_face:
        logger.info(f"  [{STAGES[3]}] {clip_name}")
        crop_params = analyze_face_positions(clip_path, target_size=target_size)
        if crop_params is None:
            logger.info(f"  → 无脸/人脸不合格淘汰: {clip_name}")
            return None, "noface"
    else:
        logger.info(f"  [跳过 {STAGES[3]}] {clip_name}")

    # ── Phase A3: 手部遮挡检测 ──
    if not skip_hand:
        logger.info(f"  [{STAGES[4]}] {clip_name}")
        is_occluded = filter_hand_occlusion_analysis(clip_path)
        if is_occluded:
            logger.info(f"  → 手部遮挡淘汰: {clip_name}")
            return None, "hand"
    else:
        logger.info(f"  [跳过 {STAGES[4]}] {clip_name}")

    # ── Phase B: 单次编码（FPS + crop + scale + audio） ──
    logger.info(f"  [{STAGES[5]}] {clip_name}")
    ready_dir = os.path.join(output_dir, "4_ready")
    os.makedirs(ready_dir, exist_ok=True)
    ready_path = os.path.join(ready_dir, clip_name)

    if os.path.exists(ready_path) and validate_video(ready_path):
        logger.info(f"  编码输出已存在且有效，跳过: {clip_name}")
    else:
        cleanup(ready_path)
        ok = encode_single_pass(
            clip_path, ready_path,
            crop_params=crop_params,
            target_fps=target_fps,
            target_size=target_size,
        )
        if not ok:
            logger.warning(f"  → 编码失败: {clip_name}")
            return None, "ffmpeg_fail"

    # ── Phase C: SyncNet 过滤 ──
    final_dir = os.path.join(output_dir, "5_final")
    if not skip_syncnet:
        logger.info(f"  [{STAGES[6]}] {clip_name}")
        final = filter_syncnet(ready_path, final_dir)

        # 清理中间文件
        if final:
            cleanup(ready_path)
            logger.info(f"  ✅ 高质量样本: {os.path.basename(final)}")
            return final, "passed"
        else:
            cleanup(ready_path)
            logger.info(f"  → SyncNet 淘汰: {clip_name}")
            return None, "syncnet"
    else:
        logger.info(f"  [跳过 {STAGES[6]}] {clip_name}")
        # 如果跳过 SyncNet，直接将 ready_path 移动到 final_dir
        os.makedirs(final_dir, exist_ok=True)
        final = os.path.join(final_dir, clip_name)
        if os.path.exists(final):
            os.remove(final)
        os.rename(ready_path, final)
        logger.info(f"  ✅ 样本已保存: {clip_name}")
        return final, "passed"


# ═══════════════════════════════════════════════════════════════
#  单个视频处理（切片 → 逐 clip）
# ═══════════════════════════════════════════════════════════════
def process_single_video(video_path, output_dir, status, stats,
                         target_fps=25, target_size=512,
                         skip_jumpcut=False, skip_face=False, skip_hand=False, skip_syncnet=False):
    """处理单个视频的完整流程"""
    video_name = os.path.basename(video_path)

    # ── 切片 ──
    logger.info(f"  [{STAGES[1]}] {video_name}")
    sliced_dir = os.path.join(output_dir, "2_sliced")
    sliced_clips = slice_video(video_path, sliced_dir)
    if not sliced_clips:
        status["videos"][video_name]["state"] = "no_valid_clips"
        return

    any_success = False

    for clip in sliced_clips:
        clip_name = os.path.basename(clip)
        stats["total_clips"] = stats.get("total_clips", 0) + 1

        # clip 级别断点续传
        clip_status = status["videos"][video_name]["clips"].get(clip_name)
        if clip_status in ["passed", "rejected"]:
            if clip_status == "passed":
                any_success = True
            continue

        try:
            result_path, reason = process_single_clip(
                clip, output_dir,
                target_fps=target_fps,
                target_size=target_size,
                skip_jumpcut=skip_jumpcut,
                skip_face=skip_face,
                skip_hand=skip_hand,
                skip_syncnet=skip_syncnet
            )

            if reason == "passed":
                status["videos"][video_name]["clips"][clip_name] = "passed"
                stats["clips_passed"] = stats.get("clips_passed", 0) + 1
                any_success = True
            else:
                status["videos"][video_name]["clips"][clip_name] = "rejected"
                key = f"clips_{reason}_reject" if reason != "ffmpeg_fail" else "clips_ffmpeg_fail"
                # 统一映射
                stat_map = {
                    "jumpcut": "clips_jumpcut_reject",
                    "noface": "clips_noface_reject",
                    "hand": "clips_hand_reject",
                    "syncnet": "clips_sync_reject",
                    "ffmpeg_fail": "clips_ffmpeg_fail",
                }
                k = stat_map.get(reason, "clips_ffmpeg_fail")
                stats[k] = stats.get(k, 0) + 1

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"  [错误] 处理 {clip_name}: {e}\n{tb}")
            status["videos"][video_name]["clips"][clip_name] = "error"

    if any_success:
        status["videos"][video_name]["state"] = "completed"
        stats["videos_with_valid_clips"] = stats.get("videos_with_valid_clips", 0) + 1
    else:
        status["videos"][video_name]["state"] = "all_rejected"
        stats["videos_all_rejected"] = stats.get("videos_all_rejected", 0) + 1


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════
def run_pipeline(input_dir, output_dir, target_fps=25, target_size=512, skip_dedup=False,
                 skip_jumpcut=False, skip_face=False, skip_hand=False, skip_syncnet=False):
    logger.info("=" * 60)
    logger.info(f"开始处理目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"目标 FPS: {target_fps}, 目标尺寸: {target_size}x{target_size}")
    logger.info(f"跳过配置: 去重={skip_dedup}, 跳切={skip_jumpcut}, 人脸={skip_face}, 手部={skip_hand}, SyncNet={skip_syncnet}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    status_file = os.path.join(output_dir, "pipeline_status.json")
    status = load_status(status_file)
    stats = status["stats"]

    # ═══ 阶段 0: MD5 去重 ═══
    dedup_dir = os.path.join(output_dir, "1_dedup")
    if skip_dedup:
        logger.info(f"\n[{STAGES[0]}] 已跳过 MD5 去重，直接使用输入目录的视频...")
        # 收集输入目录所有视频
        import glob
        unique_videos = []
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".ts"}
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    unique_videos.append(os.path.join(root, f))
    else:
        logger.info(f"\n[{STAGES[0]}] 扫描并去重...")
        unique_videos = deduplicate_videos(input_dir, dedup_dir)
        
    stats["total_videos"] = len(unique_videos)
    logger.info(f"去重后/总计有 {len(unique_videos)} 个唯一视频\n")

    # ═══ 逐视频处理 ═══
    for i, video in enumerate(unique_videos):
        video_name = os.path.basename(video)

        # 视频级断点续传
        vid_status = status.get("videos", {}).get(video_name, {})
        if vid_status.get("state") in ["completed", "all_rejected"]:
            logger.info(f"[{i+1}/{len(unique_videos)}] 跳过(已处理): {video_name}")
            continue

        logger.info(f"\n[{i+1}/{len(unique_videos)}] 正在处理: {video_name}")
        start_time = time.time()

        status.setdefault("videos", {})[video_name] = {
            "state": "processing",
            "clips": status.get("videos", {}).get(video_name, {}).get("clips", {}),
        }

        try:
            process_single_video(
                video, output_dir, status, stats,
                target_fps=target_fps,
                target_size=target_size,
                skip_jumpcut=skip_jumpcut,
                skip_face=skip_face,
                skip_hand=skip_hand,
                skip_syncnet=skip_syncnet
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"  [严重错误] {video_name}: {e}\n{tb}")
            status["videos"][video_name]["state"] = "error"

        elapsed = time.time() - start_time
        state = status["videos"][video_name].get("state", "unknown")
        logger.info(f"  耗时: {elapsed:.1f}s | 状态: {state}")

        # 定期保存
        status["stats"] = stats
        save_status(status_file, status)

    # ═══ 最终统计 ═══
    logger.info(f"\n{'='*60}")
    logger.info("处理完成！统计信息:")
    logger.info(f"  总视频数:        {stats.get('total_videos', 0)}")
    logger.info(f"  总 clip 数:      {stats.get('total_clips', 0)}")
    logger.info(f"  FFmpeg 失败:     {stats.get('clips_ffmpeg_fail', 0)}")
    logger.info(f"  跳切淘汰:        {stats.get('clips_jumpcut_reject', 0)}")
    logger.info(f"  无脸淘汰:        {stats.get('clips_noface_reject', 0)}")
    logger.info(f"  手部遮挡淘汰:    {stats.get('clips_hand_reject', 0)}")
    logger.info(f"  唇音同步淘汰:    {stats.get('clips_sync_reject', 0)}")
    logger.info(f"  ✅ 最终通过:     {stats.get('clips_passed', 0)}")
    logger.info(f"  有效视频数:      {stats.get('videos_with_valid_clips', 0)}")
    logger.info(f"  全部淘汰视频数:  {stats.get('videos_all_rejected', 0)}")
    logger.info("=" * 60)

    save_status(status_file, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频数据集预处理管道")
    parser.add_argument("--input", type=str, required=True, help="输入视频目录")
    parser.add_argument("--output", type=str, default="output_dataset", help="输出目录")
    parser.add_argument("--fps", type=int, default=25, help="目标帧率")
    parser.add_argument("--size", type=int, default=512, help="目标正方形边长")
    parser.add_argument("--skip_dedup", action="store_true", help="跳过 MD5 去重阶段")
    parser.add_argument("--skip_jumpcut", action="store_true", help="跳过 跳切检测 阶段")
    parser.add_argument("--skip_face", action="store_true", help="跳过 人脸分析 阶段")
    parser.add_argument("--skip_hand", action="store_true", help="跳过 手部遮挡检测 阶段")
    parser.add_argument("--skip_syncnet", action="store_true", help="跳过 SyncNet 过滤 阶段")
    args = parser.parse_args()
    run_pipeline(
        args.input, args.output,
        target_fps=args.fps, target_size=args.size,
        skip_dedup=args.skip_dedup,
        skip_jumpcut=args.skip_jumpcut,
        skip_face=args.skip_face,
        skip_hand=args.skip_hand,
        skip_syncnet=args.skip_syncnet
    )
