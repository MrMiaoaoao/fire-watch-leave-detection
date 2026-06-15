param(
    [string]$Python = "python",
    [switch]$SkipBaseline
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $Root
try {
    & $Python -m py_compile `
        full_pipeline.py `
        color_classifier.py `
        leave_detector.py `
        scripts\summarize_results.py
    if ($LASTEXITCODE -ne 0) {
        throw "py_compile failed with exit code $LASTEXITCODE"
    }

    & $Python -m unittest discover -s tests -p "test_*.py"
    if ($LASTEXITCODE -ne 0) {
        throw "unit tests failed with exit code $LASTEXITCODE"
    }

    if (-not $SkipBaseline) {
        & $Python scripts\summarize_results.py --strict-current
        if ($LASTEXITCODE -ne 0) {
            throw "baseline check failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
