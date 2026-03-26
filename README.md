

```markdown
# 视频数据集预处理管道

> 面向 **说话人视频生成（Talking Head Generation）** 任务的自动化数据清洗与标准化工具。  
> 从原始视频出发，经过 7 个阶段的筛选与处理，输出高质量、格式统一的训练样本。

---

## 目录

- [项目背景](#项目背景)
- [流程总览](#流程总览)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [各阶段详解](#各阶段详解)
- [目录结构](#目录结构)
- [输出说明](#输出说明)
- [配置参数](#配置参数)
- [断点续传](#断点续传)
- [性能建议](#性能建议)
- [常见问题](#常见问题)
- [技术细节](#技术细节)

---

## 项目背景

训练高质量的说话人视频生成模型（如 SadTalker、Wav2Lip、MuseTalk 等）时，
对训练数据有严格要求：

| 要求 | 原因 |
|------|------|
| 无跳切 / 镜头切换 | 保证时序连续性 |
| 人脸清晰可见且居中 | 确保模型能学到面部运动 |
| 无手部遮挡嘴部 | 嘴唇运动是核心监督信号 |
| 唇音同步准确 | 音画不同步的数据会引入噪声 |
| 统一帧率和分辨率 | 批量训练的基本要求 |
| 音频 16kHz 单声道 | wav2vec / HuBERT 特征提取要求 |

本工具将上述要求自动化，从数千个原始视频中筛选出合格样本。

---

## 流程总览

```
原始视频目录
    │
    ▼
┌─────────────────────────────────────────────────┐
│  阶段 0: MD5 去重                                │
│  · 递归扫描所有视频文件                           │
│  · 基于文件 MD5 哈希值去除完全重复的视频           │
│  · 支持增量缓存，重复运行不重新计算               │
└────────────────────┬────────────────────────────┘
                     │ 唯一视频列表
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 1: 场景切片                                │
│  · PySceneDetect 检测场景边界                     │
│  · 按场景切分为 3~60 秒的独立片段                 │
│  · 超长无切换视频按固定时长分段                    │
│  · 优先 stream copy（无损），必要时重编码          │
└────────────────────┬────────────────────────────┘
                     │ 独立 clip 列表
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 2: 跳切检测 [纯分析，不写文件]              │
│  · 光流突变检测（帧间运动量突变）                  │
│  · HSV 颜色直方图突变检测                         │
│  · AND 逻辑双重确认，减少误报                     │
│  · 连续异常帧才判定为跳切                         │
└────────────────────┬────────────────────────────┘
                     │ 通过的 clip
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 3: 人脸位置分析 [纯分析，不写文件]           │
│  · MediaPipe Face Detection 逐帧检测             │
│  · 统计人脸中心位置的中位数                       │
│  · 计算全局裁剪框（正方形，人脸居中）              │
│  · 过滤无脸帧占比过高的视频                       │
└────────────────────┬────────────────────────────┘
                     │ 裁剪参数 (x, y, w, h)
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 4: 手部遮挡检测 [纯分析，不写文件]           │
│  · MediaPipe Pose 提取手腕/指尖关键点             │
│  · 像素级距离计算（修正宽高比畸变）                │
│  · 检测手部对嘴部区域的遮挡                       │
│  · 支持早停（遮挡率明显超标时提前终止）            │
└────────────────────┬────────────────────────────┘
                     │ 通过的 clip
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 5: 单次编码输出 [唯一的有损编码步骤]         │
│  · 一条 ffmpeg 命令完成：                         │
│    - FPS 转换 → 25fps                            │
│    - 人脸居中裁剪                                 │
│    - 缩放至 512×512                              │
│    - 音频标准化 → 16kHz 单声道 AAC                │
│  · CRF 18 高质量编码                             │
│  · 编码失败自动降级（降低质量重试）                │
└────────────────────┬────────────────────────────┘
                     │ 编码后的标准化视频
                     ▼
┌─────────────────────────────────────────────────┐
│  阶段 6: SyncNet 唇音同步过滤                     │
│  · 调用 SyncNet_python 计算同步分数               │
│  · LSE-C (Confidence) ≥ 3.0                      │
│  · LSE-D (Min Distance) ≤ 10.0                   │
│  · 双条件同时满足才保留                           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
              ✅ 最终高质量样本
              输出至 5_final/
```

### 核心设计原则

**分析与编码分离**：阶段 2~4 只做帧级分析（读取像素、计算特征），不写入任何文件。
所有空间变换（FPS、裁剪、缩放、音频）合并到阶段 5 的 **一次 ffmpeg 调用** 中完成，
最大程度减少有损编码带来的质量损失。

```
传统方案: 原始 →[编码1]→ FPS →[编码2]→ 裁剪 →[编码3]→ ...  (3~4次有损编码)
本方案:   原始 →[分析]→[分析]→[分析]→[编码1]→ 最终输出      (1次有损编码)
```

---

## 环境要求

### 系统要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.8+ | 推荐 3.10 |
| FFmpeg | 4.0+ | 需在 PATH 中可用 |
| ffprobe | 随 FFmpeg 安装 | 视频信息探测 |
| CUDA | 11.0+ | SyncNet GPU 推理（可选） |

### Python 依赖

| 包 | 版本要求 | 用途 |
|---|---------|------|
| opencv-python | ≥4.5 | 视频读取、光流计算 |
| numpy | ≥1.20 | 数值计算 |
| mediapipe | 0.10.14（推荐） | 人脸检测、姿态估计 |
| scenedetect | ≥0.6 | 场景边界检测 |
| torch | ≥1.9 | SyncNet 模型推理 |

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/video-preprocess.git
cd video-preprocess
```

### 2. 创建虚拟环境

```bash
conda create -n preprocess python=3.10 -y
conda activate preprocess
```

### 3. 安装依赖

```bash
pip install opencv-python numpy mediapipe==0.10.14 scenedetect[opencv] torch
```

### 4. 验证 FFmpeg

```bash
ffmpeg -version
ffprobe -version
```

如未安装：
- **Windows**: 从 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下载，添加到 PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 5. 配置 SyncNet

```bash
# 克隆 SyncNet 仓库
git clone https://github.com/joonson/syncnet_python.git
cd syncnet_python

# 下载预训练模型
mkdir -p data
# 将 syncnet_v2.model 放到 data/ 目录下
# 模型下载地址见 SyncNet 仓库 README

# 设置环境变量
export SYNCNET_REPO=/path/to/syncnet_python     # Linux/macOS
set SYNCNET_REPO=D:\path\to\syncnet_python       # Windows CMD
$env:SYNCNET_REPO="D:\path\to\syncnet_python"    # Windows PowerShell
```

或直接修改 `preprocess_step4.py` 中的默认路径：

```python
SYNCNET_REPO = os.environ.get("SYNCNET_REPO", "/your/path/to/syncnet_python")
```

---

## 快速开始

### 基本用法

```bash
python main_pipeline.py --input /path/to/raw_videos --output /path/to/output
```

### 自定义参数

```bash
python main_pipeline.py \
    --input /data/raw_videos \
    --output /data/processed \
    --fps 25 \
    --size 512
```

### 示例

```bash
# 处理单个目录
python main_pipeline.py --input ./my_videos --output ./dataset

# 处理大型数据集
python main_pipeline.py --input /nas/video_archive --output /ssd/training_data --fps 25 --size 512
```

---

## 各阶段详解

### 阶段 0: MD5 去重 (`preprocess_step1.py`)

**目的**：消除完全相同的重复文件

**方法**：
- 递归扫描输入目录下所有视频文件（支持 `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.flv`, `.wmv`, `.ts`）
- 计算每个文件的 MD5 哈希值（64KB 分块读取）
- 相同哈希的文件只保留第一个

**缓存机制**：
- 使用 `文件绝对路径|文件大小|修改时间` 作为缓存键
- MD5 缓存保存在 `1_dedup/md5_cache.json`
- 文件未变化时直接读取缓存，避免重复计算

**输出**：不复制文件，仅返回去重后的原始路径列表

---

### 阶段 1: 场景切片 (`preprocess_step1.py`)

**目的**：将长视频按场景边界切分为独立短片段

**方法**：
1. 使用 PySceneDetect 的 `ContentDetector`（阈值 27.0）检测场景边界
2. 筛选时长在 3~60 秒之间的场景
3. 使用 `split_video_ffmpeg` 按场景边界切分

**特殊情况处理**：

| 情况 | 处理方式 |
|------|---------|
| 无场景切换 & 3~60s | 直接重封装保留（stream copy） |
| 无场景切换 & >60s | 按 60s 等分切段 |
| 无场景切换 & <3s | 丢弃 |
| 所有场景都不满足时长要求 | 丢弃 |
| 切片文件损坏 | 自动清理并重试 |

**输出**：`2_sliced/` 目录下的 `{原名}_scene_001.mp4` 等文件

---

### 阶段 2: 跳切检测 (`preprocess_step2.py`)

**目的**：过滤包含镜头跳切（但未被场景检测器捕获）的片段

**方法**：双信号 AND 确认

| 信号 | 方法 | 阈值 | 说明 |
|------|------|------|------|
| 光流突变 | Farneback 光流 | `mag_change > 25.0` | 相邻帧平均光流幅值之差 |
| 颜色突变 | HSV 直方图 Bhattacharyya 距离 | `hist_diff > 0.35` | 颜色分布相似度 |

**判定条件**：两个信号 **同时** 异常，且 **连续 2 帧** 以上 → 判定为跳切

**性能优化**：
- 每隔 2 帧采样
- 帧缩放至 25%（降低光流计算量）
- 检测到跳切立即返回（提前终止）

**输出**：无文件输出（纯分析）

---

### 阶段 3: 人脸位置分析 (`preprocess_step2.py`)

**目的**：分析人脸位置、计算裁剪参数、过滤无脸视频

**方法**：
1. MediaPipe Face Detection（model_selection=1，适合远距离人脸）逐帧检测
2. 每 3 帧采样一次
3. 取最大人脸的中心坐标和尺寸
4. 中心坐标取中位数，尺寸取 90 百分位
5. 计算正方形裁剪框：`crop_size = max(target_size, face_p90 * 2.5)`

**过滤条件**：
- 无脸帧占比 > 15% → 丢弃

**裁剪框计算规则**：
```
1. 裁剪框为正方形
2. 边长 = max(目标尺寸, 人脸90百分位尺寸 × 2.5)
3. 边长不超过视频最短边
4. 裁剪框完全在画面内（不产生黑边）
5. 人脸中心尽量居中
```

**输出**：返回 `(x, y, w, h)` 裁剪参数（无文件输出）

---

### 阶段 4: 手部遮挡检测 (`preprocess_step3.py`)

**目的**：过滤手部遮挡嘴部的视频（嘴唇运动被遮挡会严重影响训练）

**方法**：
1. MediaPipe Pose 提取 33 个身体关键点
2. 定位嘴部中心（MOUTH_LEFT 和 MOUTH_RIGHT 的中点）
3. 检测 6 个手部关键点到嘴部的距离：
   - 左/右手腕
   - 左/右食指尖
   - 左/右小指

**距离计算**：
```python
# 使用像素级距离，避免宽高比畸变
dx = point.x * frame_width - mouth_x_pixels
dy = point.y * frame_height - mouth_y_pixels
distance = sqrt(dx² + dy²)
threshold = 0.15 * frame_height  # 高度的 15%
```

**过滤条件**：
- 遮挡帧占比 > 5% → 丢弃
- 早停：已采样 10 帧以上且遮挡率 > 15% → 提前丢弃

**输出**：无文件输出（纯分析），返回 `True`（丢弃）或 `False`（通过）

---

### 阶段 5: 单次编码 (`utils.py`)

**目的**：一次性完成所有空间/时间变换，最小化质量损失

**FFmpeg 滤镜链**：
```
fps=25 → crop=W:H:X:Y → scale=512:512:flags=lanczos
```

**音频处理**：
```
AAC 128kbps → 16kHz 单声道（适配 wav2vec / HuBERT）
```

**编码参数**：
| 参数 | 值 | 说明 |
|------|---|------|
| 视频编码 | libx264 | 广泛兼容 |
| CRF | 18 | 视觉无损级别 |
| Preset | medium | 速度/质量平衡 |
| 音频编码 | AAC | 广泛兼容 |
| 音频码率 | 128kbps | |
| 音频采样率 | 16000 Hz | wav2vec 要求 |
| 声道 | 单声道 | wav2vec 要求 |

**降级策略**：常规编码失败后自动尝试
- 降低质量（CRF 23）
- 使用更快的 preset（fast）
- 增加容错参数（`genpts+discardcorrupt`）
- 单线程编码（避免并发问题）

---

### 阶段 6: SyncNet 唇音同步 (`preprocess_step4.py`)

**目的**：验证音画同步质量

**方法**：调用 [SyncNet_python](https://github.com/joonson/syncnet_python) 计算同步分数

**评估指标**：
| 指标 | 全称 | 含义 | 阈值 |
|------|------|------|------|
| LSE-C | Lip Sync Error - Confidence | 同步置信度（越高越好） | ≥ 3.0 |
| LSE-D | Lip Sync Error - Distance | 同步距离（越低越好） | ≤ 10.0 |

**判定条件**：两个指标 **同时** 满足才保留

**执行流程**：
```
1. run_pipeline.py   → 人脸追踪和裁剪
2. run_syncnet.py    → 计算同步分数
3. 解析输出文本      → 提取 Confidence 和 Min dist
```

---

## 目录结构

### 项目文件

```
video-preprocess/
├── main_pipeline.py          # 主管道入口
├── utils.py                  # 公共工具（日志、校验、编码）
├── preprocess_step1.py       # 去重 + 场景切片
├── preprocess_step2.py       # 跳切检测 + 人脸位置分析
├── preprocess_step3.py       # 手部遮挡检测
├── preprocess_step4.py       # SyncNet 唇音同步过滤
└── README.md                 # 本文件
```

### 输出目录结构

```
output_dataset/
├── 1_dedup/                  # 去重元数据
│   └── md5_cache.json        # MD5 缓存
│
├── 2_sliced/                 # 场景切片结果
│   ├── video1_scene_001.mp4
│   ├── video1_scene_002.mp4
│   └── video2_scene_001.mp4
│
├── 4_ready/                  # 编码后的标准化视频（临时，处理完自动清理）
│
├── 5_final/                  # ✅ 最终通过的高质量样本
│   ├── video1_scene_001.mp4
│   └── video2_scene_001.mp4
│
├── pipeline_status.json      # 处理状态（断点续传用）
└── preprocess.log            # 详细日志
```

---

## 输出说明

### 最终输出文件规格

| 属性 | 值 |
|------|---|
| 格式 | MP4 (H.264 + AAC) |
| 分辨率 | 512 × 512 |
| 帧率 | 25 FPS |
| 时长 | 3 ~ 60 秒 |
| 音频采样率 | 16000 Hz |
| 音频声道 | 单声道 |
| 音频码率 | 128 kbps |
| 视频 CRF | 18 |

### pipeline_status.json 格式

```json
{
  "videos": {
    "video1.mp4": {
      "state": "completed",
      "clips": {
        "video1_scene_001.mp4": "passed",
        "video1_scene_002.mp4": "rejected",
        "video1_scene_003.mp4": "passed"
      }
    },
    "video2.mp4": {
      "state": "all_rejected",
      "clips": {
        "video2_scene_001.mp4": "rejected"
      }
    }
  },
  "stats": {
    "total_videos": 100,
    "total_clips": 350,
    "clips_ffmpeg_fail": 5,
    "clips_jumpcut_reject": 30,
    "clips_noface_reject": 45,
    "clips_hand_reject": 20,
    "clips_sync_reject": 50,
    "clips_passed": 200,
    "videos_with_valid_clips": 70,
    "videos_all_rejected": 30
  }
}
```

### clip 状态值说明

| 状态 | 含义 |
|------|------|
| `passed` | 通过所有筛选，已输出到 `5_final/` |
| `rejected` | 被某个阶段淘汰 |
| `error` | 处理过程中发生异常 |

### video 状态值说明

| 状态 | 含义 |
|------|------|
| `processing` | 正在处理中 |
| `completed` | 至少有一个 clip 通过 |
| `all_rejected` | 所有 clip 都被淘汰 |
| `no_valid_clips` | 切片阶段就没有产生有效片段 |
| `error` | 视频级别的严重错误 |

---

## 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | (必填) | 输入视频目录路径 |
| `--output` | `output_dataset` | 输出目录路径 |
| `--fps` | `25` | 目标帧率 |
| `--size` | `512` | 目标正方形边长（像素） |

### 各阶段内部参数

如需修改，请直接编辑对应源文件中的默认值或函数调用参数：

#### 场景切片 (`preprocess_step1.py`)
```python
slice_video(video_path, output_dir,
            min_duration=3.0,       # 最短片段时长（秒）
            max_duration=60.0)      # 最长片段时长（秒）
```

#### 跳切检测 (`preprocess_step2.py`)
```python
detect_jump_cuts(video_path,
                 flow_sudden_threshold=25.0,  # 光流突变阈值
                 hist_threshold=0.35,         # 直方图距离阈值 (0~1)
                 min_consecutive=2,           # 连续异常帧数
                 sample_interval=2,           # 采样间隔
                 scale=0.25)                  # 帧缩放比例
```

#### 人脸分析 (`preprocess_step2.py`)
```python
analyze_face_positions(video_path,
                       target_size=512,         # 目标输出尺寸
                       max_no_face_ratio=0.15,  # 最大无脸帧比例
                       sample_interval=3)       # 采样间隔
```

#### 手部遮挡 (`preprocess_step3.py`)
```python
filter_hand_occlusion_analysis(video_path,
                                max_occlusion_ratio=0.05,  # 最大遮挡帧比例
                                sample_interval=3)         # 采样间隔
```

#### SyncNet (`preprocess_step4.py`)
```python
filter_syncnet(video_path, output_dir,
               lse_c_threshold=3.0,   # 最小置信度
               lse_d_threshold=10.0)  # 最大距离
```

#### 编码参数 (`utils.py`)
```python
encode_single_pass(input_path, output_path,
                   crop_params=None,       # (x, y, w, h) 或 None
                   target_fps=25,          # 目标帧率
                   target_size=512,        # 输出尺寸
                   crf=18,                 # 编码质量 (0=无损, 51=最差)
                   audio_rate=16000)       # 音频采样率
```

---

## 断点续传

管道内置断点续传机制，适合处理大型数据集时应对中断：

### 工作原理

1. 每处理完一个视频，自动保存状态到 `pipeline_status.json`
2. 状态文件使用 **原子写入**（先写临时文件再 `os.replace`），断电不会损坏
3. 重启时自动检测已有状态文件，跳过已完成的视频和 clip
4. 每个阶段的输出文件都会验证有效性，损坏的半成品会被自动清理

### 使用方式

```bash
# 首次运行
python main_pipeline.py --input ./videos --output ./dataset

# 中断后，直接用相同命令重新运行即可
python main_pipeline.py --input ./videos --output ./dataset
# 会自动检测到 pipeline_status.json，从断点继续
```

### 强制重新处理

```bash
# 删除状态文件即可重新开始
rm output_dataset/pipeline_status.json

# 或者只重新处理某个视频（手动编辑 JSON，将其 state 改为 "processing"）
```

---

## 性能建议

### 硬件建议

| 数据规模 | CPU | RAM | GPU | 存储 |
|---------|-----|-----|-----|------|
| < 100 视频 | 4 核 | 8 GB | 可选 | SSD 50GB |
| 100~1000 视频 | 8 核 | 16 GB | 推荐 | SSD 200GB |
| > 1000 视频 | 16+ 核 | 32 GB | 必需 | SSD 500GB+ |

### 耗时估计

| 阶段 | 每个 10s clip 耗时 | 瓶颈 |
|------|-------------------|------|
| 去重 | ~0.5s/视频 | 磁盘 IO |
| 场景切片 | ~2s | CPU (场景检测) |
| 跳切检测 | ~1s | CPU (光流) |
| 人脸分析 | ~2s | CPU/GPU (MediaPipe) |
| 手部遮挡 | ~2s | CPU/GPU (MediaPipe) |
| 编码输出 | ~3s | CPU (FFmpeg) |
| SyncNet | ~10s | GPU (PyTorch) |

**总计**：每个 10s clip 约 **20~30 秒**（含 SyncNet）

### 优化技巧

1. **SSD 存储**：避免机械硬盘成为 IO 瓶颈
2. **GPU 加速**：SyncNet 推理在 GPU 上快 5~10 倍
3. **减少不必要的步骤**：如果数据源质量较高，可以放宽阈值
4. **预先过滤**：手动删除明显不合格的视频（如纯风景、动画等）

---

## 常见问题

### Q: mediapipe 安装失败或报错 "no attribute solutions"

```bash
# 先卸载再安装指定版本
pip uninstall -y mediapipe
pip install mediapipe==0.10.14
```

> 注意：mediapipe 0.10.15+ 可能移除了 solutions API，请使用 0.10.14

### Q: FFmpeg 不在 PATH 中

```bash
# 验证
ffmpeg -version

# Windows: 下载后将 bin 目录添加到系统 PATH
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

### Q: SyncNet 分数始终为 None

检查以下几项：
1. `SYNCNET_REPO` 环境变量是否正确设置
2. `syncnet_python/data/syncnet_v2.model` 模型文件是否存在
3. SyncNet 依赖是否安装（`pip install torch torchvision`）
4. 视频是否包含音频轨（无音频的视频无法计算同步分数）

```bash
# 手动测试 SyncNet
cd /path/to/syncnet_python
python run_pipeline.py --videofile test.mp4 --reference test --data_dir /tmp/test
python run_syncnet.py --initial_model data/syncnet_v2.model --videofile test.mp4 --reference test --data_dir /tmp/test
```

### Q: 通过率太低

可以尝试放宽以下参数：

```python
# 人脸分析：允许更多无脸帧
analyze_face_positions(video_path, max_no_face_ratio=0.30)

# 手部遮挡：提高容忍度
filter_hand_occlusion_analysis(video_path, max_occlusion_ratio=0.10)

# SyncNet：降低要求
filter_syncnet(video_path, output_dir, lse_c_threshold=2.0, lse_d_threshold=12.0)
```

### Q: 通过率太高（混入低质量数据）

收紧参数：

```python
# 跳切检测：更敏感
detect_jump_cuts(video_path, flow_sudden_threshold=15.0, hist_threshold=0.25)

# SyncNet：提高要求
filter_syncnet(video_path, output_dir, lse_c_threshold=5.0, lse_d_threshold=7.0)
```

### Q: 磁盘空间不足

- 中间文件（`4_ready/`）会在每个 clip 处理完成后自动清理
- 场景切片（`2_sliced/`）在全部处理完成后可手动删除
- 最终只需保留 `5_final/` 和 `pipeline_status.json`

```bash
# 处理完成后清理中间目录
rm -rf output_dataset/2_sliced output_dataset/4_ready
```

### Q: 如何只运行部分阶段

当前版本不支持单独运行某个阶段。如果需要跳过某个阶段，可以修改 `main_pipeline.py` 中 `process_single_clip` 函数，注释掉不需要的阶段。

---

## 技术细节

### 为什么使用 AND 而非 OR 做跳切检测？

| 逻辑 | 误报场景 | 误报率 |
|------|---------|--------|
| OR | 快速转头、表情变化、环境光闪烁，任一信号触发即判定 | 高 |
| AND | 必须同时出现光流突变和颜色突变 | 低 |

正常的说话视频中，快速转头会触发光流突变但颜色分布不变；灯光变化会改变颜色但不会产生大的光流。只有真正的镜头切换才会同时满足两个条件。

### 为什么检测嘴部而非鼻子？

对于 Talking Head 任务，关键是嘴唇运动。手部遮挡鼻子但不遮挡嘴巴的情况（如擦鼻子），对训练影响有限。以嘴部中心为基准可以更精确地判断是否影响唇部可见性。

### 为什么用像素级距离而非归一化距离？

MediaPipe 输出的是归一化坐标 (0~1)。对于 16:9 视频：
- 归一化距离 0.1 在 x 方向 = 192px，在 y 方向 = 108px
- 使用归一化欧氏距离会导致水平方向的遮挡更难被检测到
- 转换为像素坐标后计算欧氏距离，阈值与方向无关

### CRF 18 够用吗？

| CRF 值 | 质量 | 适用场景 |
|--------|------|---------|
| 0 | 无损 | 存档 |
| 15 | 接近无损 | 专业后期 |
| **18** | **视觉无损** | **训练数据（推荐）** |
| 23 | 高质量 | 一般发布 |
| 28 | 中等 | 网络流媒体 |

CRF 18 在 512×512 分辨率下，人眼几乎无法察觉与原始画面的区别。
对于 AI 训练，这个质量级别足以保留人脸细节和嘴唇纹理。

---

## 许可证

MIT License

## 致谢

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - 场景检测
- [MediaPipe](https://github.com/google/mediapipe) - 人脸/姿态检测
- [SyncNet](https://github.com/joonson/syncnet_python) - 唇音同步评估
- [FFmpeg](https://ffmpeg.org/) - 视频编解码
```