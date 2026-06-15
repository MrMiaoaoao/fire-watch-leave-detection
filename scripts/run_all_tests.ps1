param(
    [string]$Python = "python",
    [string]$OutputSuffix = "",
    [int]$MaxFrames = 0,
    [string]$Device = "",
    [switch]$ClearOutputs
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutputDir = Join-Path $Root "outputs"

if ($ClearOutputs) {
    if ((Split-Path -Leaf $OutputDir) -ne "outputs") {
        throw "Refusing to clear unexpected output directory: $OutputDir"
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Get-ChildItem -LiteralPath $OutputDir -Force | Remove-Item -Recurse -Force
}

$Videos = Get-ChildItem -LiteralPath (Join-Path $Root "data") -Filter "*.mp4" |
    Sort-Object @{
        Expression = {
            if ($_.BaseName -match "(\d+)$") {
                [int]$Matches[1]
            } else {
                1
            }
        }
    }, Name

if (-not $Videos) {
    throw "No mp4 files found in data directory."
}

foreach ($Video in $Videos) {
    $VideoPath = $Video.FullName
    Write-Host "Running $VideoPath"
    $Args = @(
        (Join-Path $Root "full_pipeline.py"),
        $VideoPath,
        "--output-dir", $OutputDir,
        "--output-suffix", $OutputSuffix,
        "--max-frames", $MaxFrames
    )
    if ($Device) {
        $Args += @("--device", $Device)
    }
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline failed for $VideoPath with exit code $LASTEXITCODE"
    }
}
