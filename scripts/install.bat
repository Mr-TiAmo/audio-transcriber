@echo off
chcp 65001 >nul
setlocal
REM AudioTranscriber 一次性初始化：Python 3.12 venv + 依赖 + FFmpeg 检查 + 前端 npm install
set "ROOT=%~dp0.."

REM 1. 检查 Python 3.12
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python 3.12，请先安装：winget install -e --id Python.Python.3.12 --scope user
    exit /b 1
)

REM 2. 创建主 venv（backend\.venv）
if not exist "%ROOT%\backend\.venv\Scripts\python.exe" (
    echo [初始化] 创建虚拟环境 backend\.venv ...
    py -3.12 -m venv "%ROOT%\backend\.venv"
    if errorlevel 1 (
        echo [错误] venv 创建失败
        exit /b 1
    )
)
set "PY=%ROOT%\backend\.venv\Scripts\python.exe"

REM 3. 安装 PyTorch CUDA wheel（RTX 4060，驱动 CUDA 13.4；勿从 PyPI 装 CPU 版）
echo [依赖] 安装 PyTorch CUDA wheel（体积约 2.5GB，请耐心等待）...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [错误] PyTorch 安装失败
    exit /b 1
)

REM 4. 安装后端依赖（transformers / qwen-asr / fastapi / wetext 等）
echo [依赖] 安装后端依赖 ...
"%PY%" -m pip install -e "%ROOT%\backend[dev]"
if errorlevel 1 (
    echo [错误] 后端依赖安装失败
    exit /b 1
)

REM 5. 检查捆绑 FFmpeg（静态版，分发方式为项目内捆绑）
if not exist "%ROOT%\tools\ffmpeg\bin\ffmpeg.exe" (
    echo [警告] 未找到捆绑 FFmpeg（tools\ffmpeg\bin\ffmpeg.exe），音频解码不可用
)

REM 6. 前端依赖
where node >nul 2>&1
if errorlevel 1 (
    echo [警告] 未找到 node，跳过前端依赖安装
) else (
    if exist "%ROOT%\frontend\package.json" (
        echo [依赖] 安装前端 npm 依赖 ...
        pushd "%ROOT%\frontend"
        call npm install
        popd
    ) else (
        echo [警告] frontend 未初始化，跳过 npm install
    )
)

echo.
echo [完成] 环境初始化完成。下一步请执行 scripts\download_models.bat 下载模型。
endlocal
