@echo off
chcp 65001 >nul

echo Local Realtime Voice Translator をセットアップします。
uv venv -p 3.13
uv pip install -r requirements.txt

echo audio.cpp framework をセットアップします。
set "URL0=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.2/framework.tar.gz"
set "ZIP_FILE0=%~dp0framework.tar.gz"

set "DEST_DIR0=%~dp0"

:: curl.exe で高速取得（-L: リダイレクト追従, -o: 保存先指定）
echo Downloading llama.cpp binaries...
curl.exe -L -o "%ZIP_FILE0%" "%URL0%"

:: 解凍処理
echo Extracting files...
tar.exe -xvf "%ZIP_FILE0%"

:: zipの削除（クリーンアップ）
del "%ZIP_FILE0%"

echo audio.cpp をセットアップします。
set "URL1=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.2/audio-v0.8.2-bin-windows-x64-cuda13.3.zip"
set "ZIP_FILE1=%~dp0audio-v0.8.2-bin-windows-x64-cuda13.3.zip"
set "URL2=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.2/audio-v0.8.2-cudart-windows-x64-cuda13.3.zip"
set "ZIP_FILE2=%~dp0audio-v0.8.2-cudart-windows-x64-cuda13.3.zip"

set "DEST_DIR1=%~dp0audiocpp"

:: curl.exe で高速取得（-L: リダイレクト追従, -o: 保存先指定）
echo Downloading llama.cpp binaries...
curl.exe -L -o "%ZIP_FILE1%" "%URL1%"
curl.exe -L -o "%ZIP_FILE2%" "%URL2%"

:: 解凍処理
echo Extracting files...
powershell -Command "Expand-Archive -Path '%ZIP_FILE1%' -DestinationPath '%DEST_DIR1%' -Force"
powershell -Command "Expand-Archive -Path '%ZIP_FILE2%' -DestinationPath '%DEST_DIR1%' -Force"

:: zipの削除（クリーンアップ）
del "%ZIP_FILE1%" "%ZIP_FILE2%"

echo llama.cpp をセットアップします。
set "URL3=https://github.com/ggml-org/llama.cpp/releases/download/b11256/llama-b11256-bin-win-cuda-13.4-x64.zip"
set "ZIP_FILE3=%~dp0llama-b11256-bin-win-cuda-13.4-x64.zip"
set "URL4=https://github.com/ggml-org/llama.cpp/releases/download/b11256/cudart-llama-bin-win-cuda-13.4-x64.zip"
set "ZIP_FILE4=%~dp0cudart-llama-bin-win-cuda-13.4-x64.zip"

set "DEST_DIR2=%~dp0llamacpp"

:: curl.exe で高速取得（-L: リダイレクト追従, -o: 保存先指定）
echo Downloading llama.cpp binaries...
curl.exe -L -o "%ZIP_FILE3%" "%URL3%"
curl.exe -L -o "%ZIP_FILE4%" "%URL4%"

:: 解凍処理
echo Extracting files...
powershell -Command "Expand-Archive -Path '%ZIP_FILE3%' -DestinationPath '%DEST_DIR2%' -Force"
powershell -Command "Expand-Archive -Path '%ZIP_FILE4%' -DestinationPath '%DEST_DIR2%' -Force"

:: zipの削除（クリーンアップ）
del "%ZIP_FILE3%" "%ZIP_FILE4%"

echo Done
