@echo off
setlocal enabledelayedexpansion

REM =========================================================================
REM Llama-Fast Build Script for Windows (NVIDIA GeForce RTX 5060 / 5060 Ti)
REM Architecture: Blackwell sm_120 (Compute Capability 12.0)
REM Requires: Visual Studio 2022 (MSVC C++), CMake >= 3.28, CUDA Toolkit >= 12.8 / 13.x
REM =========================================================================

echo ============================================================
echo   Building Llama-Fast for RTX 5060 / 5060 Ti (sm_120)
echo ============================================================

set SCRIPT_DIR=%~dp0
set SRC_DIR=%SCRIPT_DIR%src
set BUILD_DIR=%SCRIPT_DIR%build-win-rtx5060

if not exist "%SRC_DIR%\CMakeLists.txt" (
    echo [ERROR] Cannot find CMakeLists.txt in %SRC_DIR%!
    exit /b 1
)

REM Verify nvcc presence
where nvcc >nul 2>nul
if errorlevel 1 (
    echo [ERROR] nvcc not found in PATH!
    echo Please make sure CUDA Toolkit 12.8+ or 13.x is installed and added to PATH.
    echo Default path is usually: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\bin"
    exit /b 1
)

echo [INFO] Configuring CMake build in %BUILD_DIR%...

cmake -B "%BUILD_DIR%" -S "%SRC_DIR%" -G "Visual Studio 17 2022" -A x64 ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="120" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=ON

if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    exit /b 1
)

echo.
echo [INFO] Compiling llama-server and llama-cli in Release mode...
cmake --build "%BUILD_DIR%" --config Release --target llama-server llama-cli llama-triattention-calibrate -j %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo.
echo ============================================================
echo   BUILD SUCCESSFUL!
echo ============================================================
echo Binaries located in: "%BUILD_DIR%\bin\Release"
echo - llama-server.exe
echo - llama-cli.exe
echo - llama-triattention-calibrate.exe
echo ============================================================

endlocal
