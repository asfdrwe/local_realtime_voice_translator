@echo off
chcp 932 > nul
echo ====================================================
echo  Git および uv の自動インストールを開始します（最終修正版）
echo ====================================================

:: 管理者権限のチェック
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] 管理者権限が必要です。
    echo 右クリックして「管理者として実行」してください。
    pause
    exit /b
)

:: 1. Gitのインストール
echo.
echo [1/2] Git をインストールしています...
winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
if %errorlevel% equ 0 (
    echo - Git のインストールが完了、または既に導入されています。
) else (
    echo [警告] Git のインストール中にエラーが発生したか、既に最新版があります。
)

:: 2. uvのインストール (正しいID: astral-sh.uv で実行)
echo.
echo [2/2] uv をインストールしています...
winget install --id astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
if %errorlevel% equ 0 (
    echo - uv のインストールが完了しました。
) else (
    echo [エラー] uv のインストールに失敗しました。
)

echo.
echo ====================================================
echo  処理が完了しました。
echo  環境変数を反映させるため、この画面を一度閉じ、
echo  新しいコマンドプロンプトを開いて確認してください。
echo ====================================================
pause
