@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "BOT_NAME=%~1"
if "%BOT_NAME%"=="" set "BOT_NAME=Daizy"

if exist "%SCRIPT_DIR%.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%SCRIPT_DIR%.env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

if "%LLM_API_URL%"=="" set "LLM_API_URL=https://api.deepseek.com/chat/completions"
if "%LLM_MODEL%"=="" set "LLM_MODEL=deepseek-chat"

python "%SCRIPT_DIR%client_llm.py" --host 127.0.0.1 --port 8888 --name "%BOT_NAME%"
