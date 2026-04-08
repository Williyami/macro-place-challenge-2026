[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Set-LocalGitConfig {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )

    & git config --local $Key $Value
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set git config '$Key'."
    }
}

& git rev-parse --show-toplevel | Tee-Object -Variable repoRoot | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not inside a git repository."
}

if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Push-Location $repoRoot
try {
    Set-LocalGitConfig "core.hooksPath" ".githooks"
    Set-LocalGitConfig "rebase.autoStash" "true"
    Set-LocalGitConfig "merge.autoStash" "true"
    Set-LocalGitConfig "fetch.prune" "true"
    Set-LocalGitConfig "rerere.enabled" "true"
    Set-LocalGitConfig "merge.ours.driver" "true"

    Write-Host "Configured local git worktree safety settings:"
    Write-Host "  core.hooksPath = .githooks"
    Write-Host "  rebase.autoStash = true"
    Write-Host "  merge.autoStash = true"
    Write-Host "  fetch.prune = true"
    Write-Host "  rerere.enabled = true"
    Write-Host "  merge.ours.driver = true"
    Write-Host ""
    Write-Host "Local pulls/rebases will now auto-stash, and pushes from a dirty tree will be blocked."
} finally {
    Pop-Location
}
