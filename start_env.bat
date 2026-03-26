@echo off
chcp 65001 >nul 2>&1

:: ─── 自动定位到脚本所在目录 ───
cd /d "%~dp0"

:: ─── 配置（相对路径） ───
set "ENV_DIR=%~dp0video_prep"
set "SYNCNET_REPO=%~dp0..\数据集\syncnet_python"

:: ─── 激活环境 ───
set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%PATH%"

echo ===================================================
echo  项目目录: %CD%
echo  Python 环境: %ENV_DIR%
python --version
echo ===================================================

:: ─── 检查关键文件 ───
if not exist "app_gradio.py" (
    echo [错误] 找不到 app_gradio.py
    echo 请将此 .bat 文件放在项目根目录下
    echo 当前目录: %CD%
    pause
    exit /b 1
)

:: ─── 快速依赖检查 ───
echo [检查依赖...]
python -c "import gradio, cv2, mediapipe, scenedetect; print('  OK')" 2>nul
if errorlevel 1 (
    echo [警告] 部分依赖缺失，尝试继续启动...
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [错误] ffmpeg 不在 PATH 中
    pause
    exit /b 1
)

:: ─── 启动 ───
echo.
echo  访问地址: http://127.0.0.1:7860
echo  按 Ctrl+C 停止
echo ===================================================
python app_gradio.py

if %errorlevel% neq 0 (
    echo.
    echo [异常退出，退出码: %errorlevel%]
    pause
)