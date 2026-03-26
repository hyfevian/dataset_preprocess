"""
Gradio 可视化界面 ─ 视频数据集预处理管道

功能：
  1. 配置参数并启动管道
  2. 实时查看处理进度和日志
  3. 浏览和预览最终输出
  4. 单个视频测试（预览各阶段效果）
"""

import os
import sys
import json
import glob
import time
import signal
import threading
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("请安装 gradio: pip install gradio")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  全局状态
# ═══════════════════════════════════════════════════════════════
class PipelineState:
    """管理管道运行状态"""

    def __init__(self):
        self.running = False
        self.process = None
        self.log_buffer = deque(maxlen=500)
        self.lock = threading.Lock()
        self.log_thread = None

    def is_running(self):
        with self.lock:
            return self.running

    def add_log(self, line):
        with self.lock:
            self.log_buffer.append(line)

    def get_logs(self):
        with self.lock:
            return "\n".join(self.log_buffer)

    def clear_logs(self):
        with self.lock:
            self.log_buffer.clear()


STATE = PipelineState()


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════
def count_videos(directory):
    """统计目录下的视频文件数量"""
    if not directory or not os.path.isdir(directory):
        return 0
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".ts"}
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count


def get_status_summary(output_dir):
    """读取 pipeline_status.json 并生成摘要"""
    status_file = os.path.join(output_dir, "pipeline_status.json")
    if not os.path.exists(status_file):
        return "尚未开始处理", {}

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            status = json.load(f)
    except Exception:
        return "状态文件读取失败", {}

    stats = status.get("stats", {})
    videos = status.get("videos", {})

    # 计算进度
    total = stats.get("total_videos", 0)
    done = sum(
        1 for v in videos.values()
        if v.get("state") in ["completed", "all_rejected", "no_valid_clips"]
    )
    processing = sum(1 for v in videos.values() if v.get("state") == "processing")
    errors = sum(1 for v in videos.values() if v.get("state") == "error")

    summary = f"""## 📊 处理进度

| 指标 | 数值 |
|------|------|
| 总视频数 | {total} |
| 已完成 | {done} |
| 处理中 | {processing} |
| 错误 | {errors} |
| **进度** | **{done}/{total} ({done/max(total,1)*100:.1f}%)** |

## 📈 Clip 级统计

| 筛选阶段 | 淘汰数 |
|----------|--------|
| FFmpeg 编码失败 | {stats.get('clips_ffmpeg_fail', 0)} |
| 跳切淘汰 | {stats.get('clips_jumpcut_reject', 0)} |
| 无脸淘汰 | {stats.get('clips_noface_reject', 0)} |
| 手部遮挡淘汰 | {stats.get('clips_hand_reject', 0)} |
| 唇音同步淘汰 | {stats.get('clips_sync_reject', 0)} |
| **✅ 最终通过** | **{stats.get('clips_passed', 0)}** |

## 📁 视频级统计

| 状态 | 数量 |
|------|------|
| 有合格 clip | {stats.get('videos_with_valid_clips', 0)} |
| 全部淘汰 | {stats.get('videos_all_rejected', 0)} |
"""
    return summary, stats


def get_final_videos(output_dir):
    """获取最终输出目录中的视频列表"""
    final_dir = os.path.join(output_dir, "5_final")
    if not os.path.isdir(final_dir):
        return []
    videos = glob.glob(os.path.join(final_dir, "*.mp4"))
    videos.sort(key=os.path.getmtime, reverse=True)
    return videos


def format_file_size(size_bytes):
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_video_thumbnail(video_path, size=(256, 256)):
    """提取视频第一帧作为缩略图"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
    except Exception:
        pass
    return np.zeros((*size, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════
#  管道控制
# ═══════════════════════════════════════════════════════════════
def start_pipeline(input_dir, output_dir, target_fps, target_size, 
                   skip_dedup, skip_jumpcut, skip_face, skip_hand, skip_syncnet):
    """启动管道（子进程方式）"""
    # 验证输入
    if not input_dir or not os.path.isdir(input_dir):
        return "❌ 输入目录不存在", STATE.get_logs()

    if STATE.is_running():
        return "⚠️ 管道已在运行中", STATE.get_logs()

    n_videos = count_videos(input_dir)
    if n_videos == 0:
        return "❌ 输入目录下没有找到视频文件", STATE.get_logs()

    STATE.clear_logs()
    STATE.add_log(f"[启动] 输入: {input_dir}")
    STATE.add_log(f"[启动] 输出: {output_dir}")
    STATE.add_log(f"[启动] 找到 {n_videos} 个视频文件")
    STATE.add_log(f"[启动] 目标 FPS={target_fps}, 尺寸={target_size}x{target_size}")
    STATE.add_log(f"[启动] 跳过配置: 去重={skip_dedup}, 跳切={skip_jumpcut}, 人脸={skip_face}, 手部={skip_hand}, SyncNet={skip_syncnet}")
    STATE.add_log("-" * 50)

    # 启动子进程
    cmd = [
        sys.executable, "main_pipeline.py",
        "--input", input_dir,
        "--output", output_dir,
        "--fps", str(int(target_fps)),
        "--size", str(int(target_size)),
    ]
    if skip_dedup:
        cmd.append("--skip_dedup")
    if skip_jumpcut:
        cmd.append("--skip_jumpcut")
    if skip_face:
        cmd.append("--skip_face")
    if skip_hand:
        cmd.append("--skip_hand")
    if skip_syncnet:
        cmd.append("--skip_syncnet")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with STATE.lock:
            STATE.running = True
            STATE.process = proc

        # 后台线程读取输出
        def _read_output():
            try:
                for line in proc.stdout:
                    line = line.rstrip("\n\r")
                    if line:
                        STATE.add_log(line)
            except Exception:
                pass
            finally:
                proc.wait()
                with STATE.lock:
                    STATE.running = False
                    STATE.process = None
                STATE.add_log(f"\n[完成] 管道已结束，退出码: {proc.returncode}")

        t = threading.Thread(target=_read_output, daemon=True)
        t.start()
        STATE.log_thread = t

        return f"✅ 管道已启动 (PID: {proc.pid})，正在处理 {n_videos} 个视频...", STATE.get_logs()

    except Exception as e:
        with STATE.lock:
            STATE.running = False
        return f"❌ 启动失败: {e}", STATE.get_logs()


def stop_pipeline():
    """停止管道"""
    with STATE.lock:
        if not STATE.running or STATE.process is None:
            return "⚠️ 管道未在运行", STATE.get_logs()

        proc = STATE.process
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            STATE.running = False
            STATE.process = None
            STATE.add_log("\n[中断] 管道已被用户停止")
        except Exception as e:
            STATE.add_log(f"\n[错误] 停止管道失败: {e}")

    return "🛑 管道已停止", STATE.get_logs()


def refresh_logs():
    """刷新日志"""
    return STATE.get_logs()


def refresh_status(output_dir):
    """刷新状态摘要"""
    if not output_dir or not os.path.isdir(output_dir):
        return "输出目录不存在"
    summary, _ = get_status_summary(output_dir)
    return summary


# ═══════════════════════════════════════════════════════════════
#  单视频测试
# ═══════════════════════════════════════════════════════════════
def test_single_video(video_path, target_size):
    """对单个视频运行各阶段分析（不写文件），返回诊断报告"""
    if not video_path or not os.path.exists(video_path):
        return "❌ 请选择一个有效的视频文件", None, None, None

    report_lines = []
    face_preview = None
    original_preview = None

    try:
        # ── 基本信息 ──
        from utils import get_video_info_ffprobe, validate_video

        report_lines.append("## 📋 基本信息\n")

        if not validate_video(video_path):
            return "❌ 视频文件损坏或不可读", None, None, None

        fps, nframes, duration = get_video_info_ffprobe(video_path)
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 原始预览帧
        ret, frame0 = cap.read()
        if ret:
            original_preview = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        cap.release()

        report_lines.append(f"| 属性 | 值 |")
        report_lines.append(f"|------|---|")
        report_lines.append(f"| 文件 | `{os.path.basename(video_path)}` |")
        report_lines.append(f"| 分辨率 | {width} × {height} |")
        report_lines.append(f"| 帧率 | {fps:.2f} FPS |")
        report_lines.append(f"| 帧数 | {nframes} |")
        report_lines.append(f"| 时长 | {duration:.2f} 秒 |")
        report_lines.append(f"| 文件大小 | {format_file_size(os.path.getsize(video_path))} |")
        report_lines.append("")

        # ── 跳切检测 ──
        report_lines.append("## 🎬 跳切检测\n")
        try:
            from preprocess_step2 import detect_jump_cuts
            has_jump = detect_jump_cuts(video_path)
            if has_jump:
                report_lines.append("❌ **检测到跳切** ─ 此视频会被淘汰\n")
            else:
                report_lines.append("✅ **未检测到跳切** ─ 通过\n")
        except Exception as e:
            report_lines.append(f"⚠️ 检测出错: {e}\n")

        # ── 人脸分析 ──
        report_lines.append("## 👤 人脸分析\n")
        try:
            from preprocess_step2 import analyze_face_positions
            target_sz = int(target_size)
            crop_params = analyze_face_positions(video_path, target_size=target_sz)
            if crop_params is None:
                report_lines.append("❌ **人脸不合格** ─ 无脸帧比例过高或无法检测到人脸\n")
            else:
                x, y, cw, ch = crop_params
                report_lines.append(f"✅ **人脸检测通过**\n")
                report_lines.append(f"| 参数 | 值 |")
                report_lines.append(f"|------|---|")
                report_lines.append(f"| 裁剪起点 | ({x}, {y}) |")
                report_lines.append(f"| 裁剪尺寸 | {cw} × {ch} |")
                report_lines.append(f"| 输出尺寸 | {target_sz} × {target_sz} |")
                report_lines.append("")

                # 生成裁剪预览
                if original_preview is not None:
                    preview = original_preview.copy()
                    # 在原图上画裁剪框（RGB）
                    cv2.rectangle(
                        preview,
                        (x, y), (x + cw, y + ch),
                        (0, 255, 0), 3
                    )
                    # 画人脸中心标记
                    cx, cy_center = x + cw // 2, y + ch // 2
                    cv2.drawMarker(
                        preview, (cx, cy_center),
                        (255, 0, 0), cv2.MARKER_CROSS, 20, 2
                    )
                    face_preview = preview

        except Exception as e:
            report_lines.append(f"⚠️ 分析出错: {e}\n")

        # ── 手部遮挡 ──
        report_lines.append("## ✋ 手部遮挡检测\n")
        try:
            from preprocess_step3 import filter_hand_occlusion_analysis
            is_occluded = filter_hand_occlusion_analysis(video_path)
            if is_occluded:
                report_lines.append("❌ **手部遮挡超标** ─ 此视频会被淘汰\n")
            else:
                report_lines.append("✅ **手部遮挡检查通过**\n")
        except Exception as e:
            report_lines.append(f"⚠️ 检测出错: {e}\n")

        # ── SyncNet（仅提示，不实际运行以节省时间） ──
        report_lines.append("## 🔊 SyncNet 唇音同步\n")
        report_lines.append("> ℹ️ 单视频测试模式下不运行 SyncNet（耗时较长）。\n")
        report_lines.append("> 如需测试，请使用完整管道或手动调用 `preprocess_step4.py`。\n")

        # ── 综合结论 ──
        report_lines.append("## 🏁 综合结论\n")
        issues = []
        if has_jump:
            issues.append("存在跳切")
        if crop_params is None:
            issues.append("人脸不合格")
        if is_occluded:
            issues.append("手部遮挡")

        if not issues:
            report_lines.append("✅ **所有分析阶段通过** ─ 此视频预计能通过管道（SyncNet 除外）\n")
        else:
            report_lines.append(f"❌ **存在以下问题**: {', '.join(issues)}\n")
            report_lines.append("此视频预计会被管道淘汰。\n")

    except Exception as e:
        import traceback
        report_lines.append(f"\n## ❌ 测试出错\n```\n{traceback.format_exc()}\n```")

    report = "\n".join(report_lines)
    return report, original_preview, face_preview, video_path


# ═══════════════════════════════════════════════════════════════
#  结果浏览
# ═══════════════════════════════════════════════════════════════
def browse_results(output_dir):
    """浏览最终输出"""
    if not output_dir:
        return "请指定输出目录", [], None

    final_videos = get_final_videos(output_dir)
    if not final_videos:
        return "暂无最终输出视���", [], None

    # 构建文件列表
    file_info = []
    total_size = 0
    for vp in final_videos:
        sz = os.path.getsize(vp)
        total_size += sz
        file_info.append(f"- `{os.path.basename(vp)}` ({format_file_size(sz)})")

    summary = f"""## 📂 最终输出 ({len(final_videos)} 个视频, 共 {format_file_size(total_size)})

{chr(10).join(file_info[:50])}
{"..." if len(file_info) > 50 else ""}
"""
    # 文件名列表给 Dropdown
    choices = [os.path.basename(v) for v in final_videos]
    return summary, choices, final_videos


def preview_result_video(video_name, output_dir):
    """预览选中的结果视频"""
    if not video_name or not output_dir:
        return None, "请选择一个视频"

    video_path = os.path.join(output_dir, "5_final", video_name)
    if not os.path.exists(video_path):
        return None, f"文件不存在: {video_path}"

    # 获取视频信息
    try:
        from utils import get_video_info_ffprobe
        fps, nframes, duration = get_video_info_ffprobe(video_path)
        info = f"""| 属性 | 值 |
|------|---|
| 文件名 | `{video_name}` |
| 帧率 | {fps:.1f} FPS |
| 帧数 | {nframes} |
| 时长 | {duration:.2f} 秒 |
| 文件大小 | {format_file_size(os.path.getsize(video_path))} |"""
    except Exception:
        info = f"文件: {video_name}"

    return video_path, info


# ═══════════════════════════════════════════════════════════════
#  Gradio 界面构建
# ═══════════════════════════════════════════════════════════════
def build_ui():
    """构建 Gradio 界面"""

    # ── 用于定时刷新的状态 ──
    _final_videos_cache = []

    with gr.Blocks(
        title="视频数据集预处理管道",
        theme=gr.themes.Soft(),
        css="""
        .log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 13px !important; }
        .status-box { min-height: 300px; }
        """
    ) as app:

        gr.Markdown("""
# 🎬 视频数据集预处理管道

面向说话人视频生成（Talking Head Generation）的自动化数据清洗工具。
从原始视频出发，经过去重、切片、跳切检测、人脸分析、手部遮挡检测、编码、唇音同步 7 个阶段，
输出高质量训练样本。
        """)

        # ═══════════════════════════════════════════════════
        #  Tab 1: 管道控制
        # ═══════════════════════════════════════════════════
        with gr.Tab("🚀 管道控制"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 参数配置")
                    input_dir = gr.Textbox(
                        label="输入目录",
                        placeholder="/path/to/raw_videos",
                        info="包含原始视频文件的目录（支持递归扫描）",
                    )
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value="output_dataset",
                        placeholder="/path/to/output",
                        info="处理结果将保存到此目录",
                    )
                    with gr.Row():
                        target_fps = gr.Number(
                            label="目标帧率",
                            value=25, minimum=1, maximum=60, step=1,
                        )
                        target_size = gr.Number(
                            label="目标尺寸",
                            value=512, minimum=128, maximum=1024, step=64,
                        )
                    with gr.Row():
                        skip_dedup = gr.Checkbox(label="跳过 MD5去重", value=False)
                        skip_jumpcut = gr.Checkbox(label="跳过 跳切检测", value=False)
                        skip_face = gr.Checkbox(label="跳过 人脸分析", value=False)
                    with gr.Row():
                        skip_hand = gr.Checkbox(label="跳过 手部遮挡", value=False)
                        skip_syncnet = gr.Checkbox(label="跳过 SyncNet", value=False)
                        
                    with gr.Row():
                        scan_btn = gr.Button("🔍 扫描输入目录", variant="secondary")
                        scan_result = gr.Textbox(
                            label="扫描结果", interactive=False, lines=1,
                        )

                    gr.Markdown("---")

                    with gr.Row():
                        start_btn = gr.Button("▶️ 启动管道", variant="primary", size="lg")
                        stop_btn = gr.Button("⏹ 停止", variant="stop", size="lg")

                    status_msg = gr.Textbox(
                        label="状态", interactive=False, lines=1,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📜 实时日志")
                    log_box = gr.Textbox(
                        label="日志输出",
                        interactive=False,
                        lines=25,
                        max_lines=25,
                        elem_classes=["log-box"],
                    )
                    with gr.Row():
                        refresh_log_btn = gr.Button("🔄 刷新日志")
                        clear_log_btn = gr.Button("🗑 清空日志")

            # ── 绑定事件 ──
            def _scan(d):
                n = count_videos(d)
                if n > 0:
                    return f"✅ 找到 {n} 个视频文件"
                elif not d:
                    return "⚠️ 请输入目录路径"
                elif not os.path.isdir(d):
                    return "❌ 目录不存在"
                else:
                    return "❌ 未找到视频文件"

            scan_btn.click(_scan, inputs=[input_dir], outputs=[scan_result])

            start_btn.click(
                start_pipeline,
                inputs=[input_dir, output_dir, target_fps, target_size, 
                        skip_dedup, skip_jumpcut, skip_face, skip_hand, skip_syncnet],
                outputs=[status_msg, log_box],
            )
            stop_btn.click(
                stop_pipeline,
                outputs=[status_msg, log_box],
            )
            refresh_log_btn.click(refresh_logs, outputs=[log_box])
            clear_log_btn.click(
                lambda: (STATE.clear_logs(), "")[1],
                outputs=[log_box],
            )

        # ═══════════════════════════════════════════════════
        #  Tab 2: 处理统计
        # ═══════════════════════════════════════════════════
        with gr.Tab("📊 处理统计"):
            gr.Markdown("### 实时处理统计")
            with gr.Row():
                stat_output_dir = gr.Textbox(
                    label="输出目录",
                    value="output_dataset",
                    scale=3,
                )
                refresh_stat_btn = gr.Button("🔄 刷新", scale=1)

            stat_display = gr.Markdown(
                value="点击「刷新」查看统计信息",
                elem_classes=["status-box"],
            )

            refresh_stat_btn.click(
                refresh_status,
                inputs=[stat_output_dir],
                outputs=[stat_display],
            )

        # ═══════════════════════════════════════════════════
        #  Tab 3: 单视频测试
        # ═══════════════════════════════════════════════════
        with gr.Tab("🧪 单视频测试"):
            gr.Markdown("""
### 单视频诊断

上传或选择一个视频文件，运行各阶段分析（不执行编码和 SyncNet），查看诊断报告。
适合调参和排查问题。
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    test_video = gr.Video(
                        label="选择测试视频",
                        sources=["upload"],
                    )
                    test_size = gr.Number(
                        label="目标尺寸", value=512,
                        minimum=128, maximum=1024, step=64,
                    )
                    test_btn = gr.Button("🔬 运行诊断", variant="primary")

                with gr.Column(scale=1):
                    test_report = gr.Markdown(
                        label="诊断报告",
                        value="上传视频后点击「运行诊断」",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 原始帧")
                    original_img = gr.Image(label="原始帧预览", height=400)
                with gr.Column():
                    gr.Markdown("#### 裁剪框预览")
                    crop_img = gr.Image(label="裁剪框（绿色）& 人脸中心（红色十字）", height=400)

            test_btn.click(
                test_single_video,
                inputs=[test_video, test_size],
                outputs=[test_report, original_img, crop_img, test_video],
            )

        # ═══════════════════════════════════════════════════
        #  Tab 4: 结果浏览
        # ═══════════════════════════════════════════════════
        with gr.Tab("📂 结果浏览"):
            gr.Markdown("### 浏览最终通过的高质量视频")

            with gr.Row():
                browse_output_dir = gr.Textbox(
                    label="输出目录",
                    value="output_dataset",
                    scale=3,
                )
                browse_btn = gr.Button("🔄 刷新文件列表", scale=1)

            browse_summary = gr.Markdown("点击「刷新文件列表」查看结果")

            with gr.Row():
                with gr.Column(scale=1):
                    video_dropdown = gr.Dropdown(
                        label="选择视频",
                        choices=[],
                        interactive=True,
                    )
                    video_info = gr.Markdown("")

                with gr.Column(scale=2):
                    result_video = gr.Video(label="视频预览")

            def _browse(d):
                summary, choices, paths = browse_results(d)
                nonlocal _final_videos_cache
                _final_videos_cache = paths
                return summary, gr.update(choices=choices, value=None)

            browse_btn.click(
                _browse,
                inputs=[browse_output_dir],
                outputs=[browse_summary, video_dropdown],
            )

            def _preview(name, d):
                path, info = preview_result_video(name, d)
                return path, info

            video_dropdown.change(
                _preview,
                inputs=[video_dropdown, browse_output_dir],
                outputs=[result_video, video_info],
            )
            
    return app

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)

