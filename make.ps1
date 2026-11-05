# Windows shim for the Makefile: .\make.ps1 <target>
# Keep the commands identical to the Makefile.
param([Parameter(Mandatory = $true)][string]$Target)
$ErrorActionPreference = "Stop"

function Run([string[]]$cmd) {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "setup"     { Run @("uv", "sync"); Run @("uv", "run", "pre-commit", "install") }
    "lint"      { Run @("uv", "run", "ruff", "check", "src", "tests", "scripts")
                  Run @("uv", "run", "ruff", "format", "--check", "src", "tests", "scripts") }
    "format"    { Run @("uv", "run", "ruff", "format", "src", "tests", "scripts")
                  Run @("uv", "run", "ruff", "check", "--fix", "src", "tests", "scripts") }
    "test"      { Run @("uv", "run", "pytest", "-m", "not slow") }
    "demo-stub" { Run @("uv", "run", "python", "scripts/demo_stub.py") }
    "serve"     { Run @("uv", "run", "python", "scripts/mock_server.py") }
    default     { Write-Error "Unknown target '$Target'. Use: setup, lint, format, test, demo-stub, serve" }
}
