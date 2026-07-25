param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MemopilotArgs
)

$ErrorActionPreference = "Stop"
$prototypeRoot = "D:\PROJECTS\MemoPilot-Agent-Main"
$prototypeConfig = Join-Path $prototypeRoot "config.toml"
$prototypeMcp = "C:\Users\20618\.memopilot\workspace\mcp_servers.json"
$workspace = Join-Path $PSScriptRoot "..\workspace"

if (-not (Test-Path -LiteralPath $prototypeConfig)) {
    throw "找不到旧原型配置: $prototypeConfig"
}
if (-not (Test-Path -LiteralPath $prototypeMcp)) {
    throw "找不到旧原型 MCP 配置: $prototypeMcp"
}

$mcp = Get-Content -Raw -LiteralPath $prototypeMcp | ConvertFrom-Json
$steam = $mcp.servers.steam
if ($null -eq $steam -or $null -eq $steam.env) {
    throw "旧原型 MCP 配置缺少 steam 环境变量"
}

# 密钥只进入当前 PowerShell 进程，不写入 MemoPilot 仓库。
$env:STEAM_API_KEY = [string]$steam.env.STEAM_API_KEY
$env:STEAM_ID = [string]$steam.env.STEAM_ID
# 旧原型未声明 text-embedding-v2 的维度；DashScope 该模型使用 1536 维。
$env:MEMOPILOT_EMBEDDING_DIMENSION = "1536"

& uv run memopilot @MemopilotArgs --config $prototypeConfig --workspace $workspace
exit $LASTEXITCODE
