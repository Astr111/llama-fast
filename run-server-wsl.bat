@echo off
setlocal enabledelayedexpansion

REM =========================================================================
REM Llama-Fast WSL Launcher for Windows
REM Automatically executes the Linux CUDA build inside your default WSL distro
REM =========================================================================

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Convert Windows path to WSL path
for /f "usebackq tokens=*" %%i in (`wsl wslpath -u "%SCRIPT_DIR%"`) do set WSL_REPO_DIR=%%i

if "%WSL_REPO_DIR%"=="" (
    echo [ERROR] Failed to resolve WSL path. Please verify WSL is installed and running.
    exit /b 1
)

echo [INFO] WSL Repository Path: %WSL_REPO_DIR%
echo [INFO] Launching Llama-Fast Server under CUDA 13...

REM If arguments were passed to this .bat file, forward them.
REM Otherwise, use the recommended default launch parameters.
if "%~1"=="" (
    set LAUNCH_CMD=cd '%WSL_REPO_DIR%' && ./dist/v1.0.1/cuda13/run-server.sh \
      -m /path/to/Ternary-Bonsai-4B-Q2_0_g64.gguf \
      -ngl 99 \
      -c 32768 \
      -np 1 \
      -ctk q8_0 \
      -ctv turbo3 \
      --chat-template chatml \
      --logit-bias 151657-inf,151658-inf \
      --no-cache-prompt \
      --triattention-stats calibration/bonsai-4b.triattention \
      --triattention-budget 2048 \
      --triattention-window 512 \
      --triattention-offset-max 4096 \
      --triattention-normalize \
      --triattention-protect-prefill \
      --port 8080 \
      --host 0.0.0.0
) else (
    set LAUNCH_CMD=cd '%WSL_REPO_DIR%' && ./dist/v1.0.1/cuda13/run-server.sh %*
)

wsl -e bash -c "!LAUNCH_CMD!"

endlocal
