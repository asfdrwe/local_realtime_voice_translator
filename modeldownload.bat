@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ==========================================
:: 設定エリア（保存先フォルダを指定してください）
:: ==========================================
set "SAVE_DIR=%~dp0models"

:: ==========================================
:: メイン処理
:: ==========================================
echo ==================================================
echo AIモデルの事前ダウンロードを開始します...
echo 保存先: %SAVE_DIR%
echo ==================================================

:: 1. Qwen3-ASR-0.6B-Q8_0.gguf のダウンロード
echo.
echo [1/2] Qwen3-ASR-0.6B-Q8_0.gguf をダウンロード中...
set "QWEN3_ASR_URL=https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf?download=true"
set "QWEN3_ASR_FILE=%SAVE_DIR%\qwen3-asr-0.6b-q8_0.gguf"

if exist "%QWEN3_ASR_FILE%" (
    echo すでにファイルが存在するためスキップします。
) else (
    curl.exe -L -o "%QWEN3_ASR_FILE%" "%QWEN3_ASR_URL%"
    if !errorlevel! neq 0 (
        echo [エラー] Qwen3 ASRのダウンロードに失敗しました。
        goto :error
    )
)

:: 2. Qwen3-TTS-12Hz-0.6B-base-Q8_0.gguf のダウンロード
echo.
echo [1/2] Qwen3-TTS-12Hz-0.6B-base-Q8_0.ggufをダウンロード中...
set "QWEN3_TTS_URL=https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf?download=true"
set "QWEN3_TTS_FILE=%SAVE_DIR%\qwen3-tts-12hz-0.6b-base-q8_0.gguf"

if exist "%QWEN3_TTS_FILE%" (
    echo すでにファイルが存在するためスキップします。
) else (
    curl.exe -L -o "%QWEN3_TTS_FILE%" "%QWEN3_TTS_URL%"
    if !errorlevel! neq 0 (
        echo [エラー] Qwen3 TTSのダウンロードに失敗しました。
        goto :error
    )
)

:: 3. Gemma-4-E4B-it-Q4_K_M.gguf のダウンロード
echo.
echo [1/2] Gemma-4-E4B-it-Q4_K_M.gguf をダウンロード中...
set "GEMMA_URL=https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true"
set "GEMMA_FILE=%SAVE_DIR%\gemma-4-E4B-it-Q4_K_M.gguf"

if exist "%GEMMA_FILE%" (
    echo すでにファイルが存在するためスキップします。
) else (
    curl.exe -L -o "%GEMMA_FILE%" "%GEMMA_URL%"
    if !errorlevel! neq 0 (
        echo [エラー] Gemmaのダウンロードに失敗しました。
        goto :error
    )
)

echo ===================================================
echo [O] すべてのモデルの事前ダウンロードが完了しました！
echo ===================================================
pause
exit /b 0

:error
echo ===================================================
echo [X] エラーが発生したため処理を中断しました。
echo ===================================================
pause
exit /b 1
