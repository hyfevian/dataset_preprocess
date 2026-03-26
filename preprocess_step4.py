"""
阶段 4 ─ SyncNet 唇音同步过滤

强制使用 SyncNet 进行唇音同步验证。
"""

import os
import re
import shutil
import subprocess
import tempfile
import logging

from utils import cleanup

logger = logging.getLogger("pipeline")

# ── SyncNet 仓库路径 ──
# 优先使用环境变量，默认回退到当前脚本所在目录下的 syncnet_python 文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNCNET_REPO = os.environ.get(
    "SYNCNET_REPO",
    os.path.join(BASE_DIR, "syncnet_python")
)


# ═══════════════════════════════════════════════════════════════
#  对外接口
# ═══════════════════════════════════════════════════════════════
def filter_syncnet(
    video_path,
    output_dir,
    lse_c_threshold=3.0,
    lse_d_threshold=10.0,
):
    """
    LSE-C ≥ lse_c_threshold 且 LSE-D ≤ lse_d_threshold → 保留。

    Returns
    -------
    str : 输出文件路径（通过）
    None : 未通过
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, base_name)

    if os.path.exists(output_path):
        from utils import validate_video
        if validate_video(output_path):
            return output_path
        else:
            cleanup(output_path)

    lse_c, lse_d = get_syncnet_scores(video_path)

    if lse_c is None or lse_d is None:
        logger.error(f"  无法获取 SyncNet 分数，强制丢弃: {video_path}")
        return None

    logger.info(f"  SyncNet: LSE-C={lse_c:.2f}  LSE-D={lse_d:.2f}")

    passed = lse_c >= lse_c_threshold and lse_d <= lse_d_threshold
    if passed:
        logger.info("  → ✅ 保留")
        shutil.copy2(video_path, output_path)
        return output_path
    else:
        reasons = []
        if lse_c < lse_c_threshold:
            reasons.append(f"LSE-C {lse_c:.2f} < {lse_c_threshold}")
        if lse_d > lse_d_threshold:
            reasons.append(f"LSE-D {lse_d:.2f} > {lse_d_threshold}")
        logger.info(f"  → ❌ 淘汰 ({', '.join(reasons)})")
        return None


# ═══════════════════════════════════════════════════════════════
#  分数获取
# ═══════════════════════════════════════════════════════════════
def get_syncnet_scores(video_path):
    """
    返回 (lse_c, lse_d)。
    优先 CLI，其次 API，都失败返回 (None, None)。
    """
    # Level-1: CLI
    if SYNCNET_REPO and os.path.isdir(SYNCNET_REPO):
        result = _scores_cli(video_path)
        if result[0] is not None:
            return result

    # Level-2: Python API (预留)
    result = _scores_api(video_path)
    if result[0] is not None:
        return result

    return None, None


def _scores_cli(video_path):
    """调用 SyncNet_python 命令行"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = os.path.splitext(os.path.basename(video_path))[0]

            # 1. 人脸追踪和裁剪
            cmd_pipeline = [
                "python",
                os.path.join(SYNCNET_REPO, "run_pipeline.py"),
                "--videofile", video_path,
                "--reference", ref,
                "--data_dir", tmpdir,
            ]
            subprocess.run(
                cmd_pipeline,
                capture_output=True, text=True,
                encoding="utf-8", errors="ignore",
                timeout=300,
            )

            # 2. SyncNet 评估
            cmd_syncnet = [
                "python",
                os.path.join(SYNCNET_REPO, "run_syncnet.py"),
                "--initial_model",
                os.path.join(SYNCNET_REPO, "data", "syncnet_v2.model"),
                "--videofile", video_path,
                "--reference", ref,
                "--data_dir", tmpdir,
            ]
            r = subprocess.run(
                cmd_syncnet,
                capture_output=True, text=True,
                encoding="utf-8", errors="ignore",
                timeout=300,
            )
            txt = r.stdout + r.stderr

            # 解析
            c = _parse(txt, r"Confidence:\s+([0-9.eE+-]+)")
            d = _parse(txt, r"Min dist:\s+([0-9.eE+-]+)")

            # 兼容其他格式
            if c is None:
                c = _parse(txt, r"LSE[_-]?C[:\s]+([0-9.eE+-]+)")
            if d is None:
                d = _parse(txt, r"LSE[_-]?D[:\s]+([0-9.eE+-]+)")

            return c, d
    except subprocess.TimeoutExpired:
        logger.warning(f"  SyncNet CLI 超时: {video_path}")
        return None, None
    except Exception as e:
        logger.warning(f"  SyncNet CLI 错误: {e}")
        return None, None


def _scores_api(video_path):
    """预留的 Python API 入口"""
    # TODO: 实现 SyncNet Python API 调用
    # from your_syncnet_module import evaluate
    # return evaluate(video_path)
    return None, None


def _parse(text, pattern):
    """从文本中解析浮点数"""
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None
