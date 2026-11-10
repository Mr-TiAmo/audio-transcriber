@echo off
chcp 65001 >nul
setlocal
REM AudioTranscriber 启动脚本（V3.0 设计 §13.1：检查 -> 启动 -> 等待 /health -> 打开浏览器）
set "ROOT=%~dp0.."
set "PY=%ROOT%\backend\.venv\Scripts\python.exe"
set "PORT=8730"

REM 1. 基础检查
py -3.12 --version >nul 2>&1
if not exist "%PY%" (
    echo [错误] 未找到虚拟环境，请先执行 scripts\install.bat
    exit /b 1
)
if not exist "%ROOT%\tools\ffmpeg\bin\ffmpeg.exe" (
    echo [警告] 未找到捆绑 FFmpeg，音频解码将不可用
)

REM 2. 捆绑 FFmpeg 加入本次 PATH（不污染系统 PATH）
set "PATH=%ROOT%\tools\ffmpeg\bin;%PATH%"

REM 3. 端口占用检查（被本应用占用则提示）
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [警告] 端口 %PORT% 已被占用，若为旧实例请先执行 scripts\stop.bat
)

REM 4. 启动 FastAPI（单进程、单 Uvicorn Worker，监听 127.0.0.1）
echo [启动] FastAPI 本地服务 http://127.0.0.1:%PORT% ...
start "AudioTranscriber-API" /min cmd /c ""%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% --app-dir "%ROOT%\backend\src""

REM 5. 等待 /health 成功（最多 60 秒）
set /a tries=0
:wait_health
timeout /t 2 /nobreak >nul
curl -s -o nul http://127.0.0.1:%PORT%/health 2>nul
if not errorlevel 1 goto health_ok
set /a tries+=1
if %tries% lss 30 goto wait_health
echo [错误] 服务未能在 60 秒内就绪，请查看 logs 目录
exit /b 1
:health_ok
echo [就绪] 服务健康检查通过

REM 6. 打开默认浏览器
start "" http://127.0.0.1:%PORT%

echo [完成] 应用已启动。关闭请执行 scripts\stop.bat
endlocal
