@echo off
chcp 65001 >nul
setlocal
REM 下载本地模型（一次性初始化；走 hf-mirror.com 国内镜像，设计 §13.3）
REM 下载完成后推理链路离线运行（allow_download_during_transcribe = false）
set "ROOT=%~dp0.."
set "PY=%ROOT%\backend\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [错误] 未找到虚拟环境，请先执行 scripts\install.bat
    exit /b 1
)

REM 使用国内镜像（可用环境变量覆盖；直连 huggingface.co 可行时不设 HF_ENDPOINT）
if "%HF_ENDPOINT%"=="" set "HF_ENDPOINT=https://hf-mirror.com"

REM 注意：新版 huggingface_hub 的 CLI 入口是 hf（python -m huggingface.cli 已移除）
set "HF=%ROOT%\backend\.venv\Scripts\hf.exe"

REM 默认引擎：Qwen3-ASR-0.6B（V3.0 实测性价比最优）
echo [下载] Qwen/Qwen3-ASR-0.6B -^> models\Qwen3-ASR-0.6B ...
"%HF%" download Qwen/Qwen3-ASR-0.6B --local-dir "%ROOT%\models\Qwen3-ASR-0.6B"
if errorlevel 1 (
    echo [提示] 镜像下载失败，尝试 huggingface.co 直连 ...
    "%HF%" download Qwen/Qwen3-ASR-0.6B --local-dir "%ROOT%\models\Qwen3-ASR-0.6B"
)
if errorlevel 1 (
    echo [错误] 模型下载失败，请检查网络或 HF_ENDPOINT 配置
    exit /b 1
)

echo.
echo [可选] 质量优先引擎（1.7B，工作台任务栏下拉可选）：
echo   "%HF%" download Qwen/Qwen3-ASR-1.7B --local-dir "%ROOT%\models\Qwen3-ASR-1.7B"
echo   "%PY%" "%ROOT%\backend\scripts\prepare_model_dir.py" "%ROOT%\models\Qwen3-ASR-1.7B"
echo [可选] 语义纠错模型（默认关闭的后处理高级选项）：
echo   "%HF%" download twnlp/ChineseErrorCorrector4-4B --local-dir "%ROOT%\models\ChineseErrorCorrector4-4B"
echo.
echo [完成] 默认模型已就绪，可执行 scripts\start.bat
endlocal
