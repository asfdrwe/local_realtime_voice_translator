@echo off
chcp 65001 >nul

:: Windows Terminalを起動
:: 各タブの cmd /k 内で直接 set を実行して環境変数を確実に引き渡します
start "" wt ^
  -d "%~dp0." --title "LLM" cmd /k ".\llamacpp\llama-server.exe -m .\models\gemma-4-E4B-it-Q4_K_M.gguf -c 4096" ; ^
  -d "%~dp0." --title "AUDIO" cmd /k ".\audiocpp\audiocpp_server.exe --config .\voicetranslator.json" ; ^
  -d "%~dp0." --title "MAIN" cmd /k "uv run python main.py"
