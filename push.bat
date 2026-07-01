@echo off
cd /d "%~dp0"

set /p MSG="コミットメッセージ: "
if "%MSG%"=="" (
    echo メッセージが空です。中止します。
    pause
    exit /b 1
)

git add -A
git diff --cached --name-only
git commit -m "%MSG%"
git push origin main

echo.
echo Push 完了: https://github.com/rrpiyo-bit/paint-pola
pause
