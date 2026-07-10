@echo off
REM ============================================================
REM  Gephi SSH Tunnel — Nanite KG 实时可视化
REM  本机 (Windows) 到服务器 (cn.hi-cat.top) 的反向隧道
REM
REM  用法:
REM    1. 打开 Gephi → Tools → Streaming → Server → Start
REM    2. 双击运行本脚本
REM    3. 保持窗口开启，不要关闭
REM
REM  原理:
REM    服务器 localhost:8080  →  SSH 隧道  →  本机 Gephi:8080
REM ============================================================

setlocal

set SSH_HOST=cn.hi-cat.top
set SSH_PORT=52222
set SSH_USER=hjq
set LOCAL_PORT=8080
set REMOTE_PORT=8080

echo ════════════════════════════════════════════
echo  Gephi SSH Tunnel — Nanite KG Streaming
echo ════════════════════════════════════════════
echo.
echo  [1/3] 检查 SSH 连接...
where ssh >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  ❌ SSH not found! 请安装 OpenSSH 客户端。
    pause
    exit /b 1
)
echo  ✅ SSH 可用
echo.
echo  [2/3] 检查 Gephi 端口...
netstat -an | find ":%LOCAL_PORT% " >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo  ✅ 端口 %LOCAL_PORT% 已监听 (Gephi Streaming 运行中)
) else (
    echo  ⚠️  端口 %LOCAL_PORT% 未检测到监听。
    echo      请确认 Gephi → Tools → Streaming → Server → Start
    echo      按任意键继续...
    pause >nul
)
echo.
echo  [3/3] 建立反向隧道...
echo.
echo  服务器 :%REMOTE_PORT% → 本机 localhost:%LOCAL_PORT%
echo.
echo  ⏳ 正在连接 %SSH_HOST%:%SSH_PORT% ...
echo  (如提示输入密码，请输入服务器密码)
echo  窗口关闭后隧道即断开。
echo.

ssh -p %SSH_PORT% -R %REMOTE_PORT%:localhost:%LOCAL_PORT% %SSH_USER%@%SSH_HOST%

echo.
echo  ⛔ 隧道已关闭
pause
