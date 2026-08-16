<#
.SYNOPSIS
    Backs up the FastNoise2 binaries vendored into the PorismDIMsWorldGen plugin.

.DESCRIPTION
    Run this before any rebuild that overwrites the 8 vendored binaries.

    Writes a SHA256 manifest next to the copies so a later restore can be
    verified byte-for-byte.

    The backup lands OUTSIDE the repository, and that is not a preference.
    BuildPlugin copies /Source/ wholesale into the package Epic rebuilds from,
    so a folder of superseded libraries parked next to the live ones ships to
    customers and can be linked by accident - it happened once, 91 MB of
    FastNoise v0.10 binaries sitting in Source/ThirdParty/ ready to travel.
    Pass -BackupRoot to choose your own location; anything inside the plugin is
    the one answer that is wrong.
#>
[CmdletBinding()]
param(
    [string] $PluginLibs = 'D:\GIT\PorismDIMsWorldGen\Plugins\PorismDIMsWorldGenerator\Source\ThirdParty\FastNoise2\libs',
    [string] $BackupRoot,
    [int]    $ExpectedBinaryCount = 8
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $PluginLibs)) { throw "Plugin libs folder not found: $PluginLibs" }

if (-not $BackupRoot) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

    # Climb to the repository root and place the backup BESIDE it, so the copy
    # is outside the tree BuildPlugin packages no matter where the repo lives.
    $repoRoot = Split-Path $PluginLibs -Parent
    while ($repoRoot -and -not (Test-Path (Join-Path $repoRoot '.git'))) {
        $parent = Split-Path $repoRoot -Parent
        if ($parent -eq $repoRoot) { $repoRoot = $null; break }
        $repoRoot = $parent
    }
    if (-not $repoRoot) {
        throw ("Could not find the repository root above '$PluginLibs', so the backup location " +
               "cannot be derived. Pass -BackupRoot explicitly, and keep it outside the plugin.")
    }

    $BackupRoot = Join-Path (Split-Path $repoRoot -Parent) "_PorismLibsBackup\libs-backup-$stamp"
}

# Cheap, and it catches the mistake this script exists to prevent - including a
# caller who passes -BackupRoot by hand.
$pluginRoot = (Resolve-Path (Join-Path $PluginLibs '..\..\..\..')).Path
if ($BackupRoot -like (Join-Path $pluginRoot '*')) {
    throw ("Refusing to write the backup to '$BackupRoot': it is inside the plugin, and BuildPlugin " +
           "would ship those superseded binaries to customers. Choose a location outside '$pluginRoot'.")
}

Write-Host "Source : $PluginLibs"
Write-Host "Backup : $BackupRoot"

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null

$binaryExt = @('.dll', '.lib', '.so', '.a')
$binaries = Get-ChildItem -Path $PluginLibs -Recurse -File |
    Where-Object { $binaryExt -contains $_.Extension }

if ($binaries.Count -ne $ExpectedBinaryCount) {
    throw ("Expected $ExpectedBinaryCount binaries under '$PluginLibs' but found $($binaries.Count). " +
           "Refusing to back up a layout that does not match the known-good one.")
}

Copy-Item -Path (Join-Path $PluginLibs '*') -Destination $BackupRoot -Recurse -Force

$manifest = Join-Path $BackupRoot 'manifest.txt'
$lines = foreach ($b in $binaries) {
    $rel = $b.FullName.Substring($PluginLibs.Length).TrimStart('\')
    $hash = (Get-FileHash -Path $b.FullName -Algorithm SHA256).Hash
    '{0}  {1}  {2}' -f $hash, $b.Length, $rel
}
$lines | Out-File -FilePath $manifest -Encoding utf8
$lines | ForEach-Object { Write-Host "  $_" }

# The live binaries are meant to be COMMITTED - they are build input a fresh
# clone cannot produce. This warns if a .gitignore change has silently taken
# them back out, which would leave this backup as their only copy again.
Push-Location (Split-Path $PluginLibs -Parent)
try {
    $probe = $binaries[0].FullName
    & git check-ignore -q -- $probe
    if ($LASTEXITCODE -eq 0) {
        Write-Warning ("git IGNORES '$probe' - the vendored FastNoise2 binaries are no longer tracked, " +
                       "so this backup is now the only copy. Restore the '!.../FastNoise2/libs/**' rules in .gitignore.")
    }
}
catch { Write-Warning "Could not run 'git check-ignore': $_" }
finally { Pop-Location }

Write-Host ""
Write-Host "Backed up $($binaries.Count) binaries."
Write-Host "Manifest: $manifest"

# 'git check-ignore' answers by exit code: 1 means "not ignored", which is the
# outcome we want and must not leak out as this script's own failure.
exit 0
