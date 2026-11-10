@echo off
chcp 65001 >nul
setlocal
REM 停止 AudioTranscriber：终止 API 及其子进程树（编排层 Worker 会随 API 生命周期清理）
set "PORT=8730"

echo [停止] 查找占用端口 %PORT% 的进程 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo [停止] 终止进程树 PID=%%p ...
    taskkill /pid %%p /t /f >nul 2>&1
)
echo [完成] 已停止
endlocal
