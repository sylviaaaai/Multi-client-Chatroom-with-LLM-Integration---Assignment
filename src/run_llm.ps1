param(
    [string]$Name = "Daizy",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8888
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvPath = Join-Path $ScriptDir ".env"

if (Test-Path $EnvPath) {
    Get-Content $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $namePart, $valuePart = $line.Split("=", 2)
        $namePart = $namePart.Trim()
        $valuePart = $valuePart.Trim().Trim('"').Trim("'")

        if ($namePart) {
            Set-Item -Path "Env:$namePart" -Value $valuePart
        }
    }
}

if (-not $env:LLM_API_URL) {
    $env:LLM_API_URL = "https://api.deepseek.com/chat/completions"
}

if (-not $env:LLM_MODEL) {
    $env:LLM_MODEL = "deepseek-chat"
}

python (Join-Path $ScriptDir "client_llm.py") --host $Host --port $Port --name $Name
