@echo off
setlocal enabledelayedexpansion

:: 设置窗口标题和颜色
title Lumina Command Center
color 0B
cd /d "%~dp0"

cls
echo.
echo ========================================================
echo.
echo      /$$                               /$$                    
echo     ^| $$                              ^|__/                    
echo     ^| $$      /$$   /$$ /$$$$$$/$$$$  /$$ /$$$$$$$   /$$$$$$ 
echo     ^| $$     ^| $$  ^| $$^| $$_  $$_  $$ ^| $$^| $$__  $$ ^|____  $$
echo     ^| $$     ^| $$  ^| $$^| $$ \ $$ \ $$ ^| $$^| $$  \ $$  /$$$$$$$
echo     ^| $$     ^| $$  ^| $$^| $$ ^| $$ ^| $$ ^| $$^| $$  ^| $$ /$$__  $$
echo     ^| $$$$$$$^|  $$$$$$/^| $$ ^| $$ ^| $$ ^| $$^| $$  ^| $$^|  $$$$$$$
echo     ^|________/ \______/ ^|__/ ^|__/ ^|__/ ^|__/^|__/  ^|__/ \_______/
echo.
echo ========================================================
echo          LUMINA TEMPLE - DUAL TOWER SYSTEM
echo ========================================================
echo.

:: 1. 激活环境
if exist "venv\Scripts\activate.bat" (
    echo  [*] Activating venv...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo  [*] Activating .venv...
    call .venv\Scripts\activate.bat
) else (
    echo  [!] Using system Python...
)

:: 2. 检查并安装 MCP 依赖
python -c "import mcp" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [!] Installing dependencies for MCP...
    pip install "mcp[cli]" httpx uvicorn fastapi sqlmodel
    echo  [*] Dependencies installed.
    echo.
)

echo.
echo  --------------------------------------------------------
echo  [SYSTEM LAUNCH SEQUENCE]
echo  --------------------------------------------------------
echo.

:: 3. 启动主程序 (Web Tower) - 端口 8000
echo  [1/2] Launching Web Core (main.py) on Port 8000...
start "Lumina Web Core (Do Not Close)" cmd /k "python main.py"

:: 等待 3 秒让主程序先跑起来
timeout /t 3 /nobreak >nul

:: 4. 启动 MCP 服务 (MCP Tower) - 端口 8001
echo  [2/2] Launching MCP Server (mcp_server.py) on Port 8001...
start "Lumina MCP Agent (SSE Mode)" cmd /k "python mcp_server.py"

echo.
echo ========================================================
echo  SYSTEM ONLINE
echo  - Web Interface: http://localhost:8000/generate.html
echo  - MCP SSE URL:   http://localhost:8001/sse
echo ========================================================
echo.
echo  You can now configure your AI client (Cursor/Claude) 
echo  to connect to the MCP URL above.
echo.
pause