<#
.SYNOPSIS
    Builds FastNoise.dll / FastNoise.lib (Win64, Release) from this fork.

.DESCRIPTION
    Configures with Ninja inside a Visual Studio x64 developer environment.
    FastSIMD is taken from a local clone via CPM_FastSIMD_SOURCE so the build
    needs no network and always uses the commit FastNoise2 pins.
#>
[CmdletBinding()]
param(
    [string] $SourceDir   = 'D:\GIT\FastNoise2Fork-v111',
    [string] $BuildDir,
    [string] $FastSimdDir = 'D:\GIT\FastSIMD-pin',
    [string] $VsInstallDir,
    [string] $NinjaPath,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config = 'Release'
)

$ErrorActionPreference = 'Stop'
if (-not $BuildDir) { $BuildDir = Join-Path $SourceDir 'out\build\win64' }

if (-not (Test-Path (Join-Path $SourceDir 'CMakeLists.txt'))) { throw "No CMakeLists.txt in $SourceDir" }
if (-not (Test-Path (Join-Path $FastSimdDir 'CMakeLists.txt'))) { throw "No FastSIMD source in $FastSimdDir" }

# --- Visual Studio -----------------------------------------------------------
if (-not $VsInstallDir) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; pass -VsInstallDir explicitly." }
    # Prefer VS2022 (17.x) - that is what produced the shipped binaries.
    $VsInstallDir = & $vswhere -version '[17.0,18.0)' -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $VsInstallDir) {
        $VsInstallDir = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        Write-Warning "No VS2022 found, falling back to: $VsInstallDir"
    }
}
if (-not (Test-Path $VsInstallDir)) { throw "Visual Studio not found: $VsInstallDir" }

if (-not $NinjaPath) {
    $candidate = Join-Path $VsInstallDir 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
    if (Test-Path $candidate) { $NinjaPath = $candidate }
    else {
        $onPath = Get-Command ninja -ErrorAction SilentlyContinue
        if ($onPath) { $NinjaPath = $onPath.Source } else { throw "ninja.exe not found; pass -NinjaPath." }
    }
}

Write-Host "PHASE :: Win64 configure"
Write-Host "  Source   : $SourceDir"
Write-Host "  Build    : $BuildDir"
Write-Host "  FastSIMD : $FastSimdDir"
Write-Host "  VS       : $VsInstallDir"
Write-Host "  ninja    : $NinjaPath"

# Import the VS x64 toolchain environment into this session.
# vcvars64.bat, not VsDevCmd.bat: the latter shells out to vswhere.exe, which is
# not resolvable from every VS layout on this machine.
$vcvars = Join-Path $VsInstallDir 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }
# stderr is swallowed inside cmd on purpose: vcvars64 emits benign warnings there,
# and PowerShell 5.1 turns any native stderr line into a terminating error.
$envDump = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed ($LASTEXITCODE)" }
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue }
}
$clProbe = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $clProbe) { throw "cl.exe still not on PATH after vcvars64.bat" }
Write-Host "  cl.exe   : $($clProbe.Source)"

# CMake re-emits these paths into generated CMake code, where a backslash is an
# escape character - always hand it forward slashes.
$fwd = { param($p) $p -replace '\\', '/' }

$cmakeArgs = @(
    '-S', (& $fwd $SourceDir)
    '-B', (& $fwd $BuildDir)
    '-G', 'Ninja'
    "-DCMAKE_MAKE_PROGRAM=$(& $fwd $NinjaPath)"
    "-DCMAKE_BUILD_TYPE=$Config"
    '-DBUILD_SHARED_LIBS=ON'
    '-DFASTNOISE2_TOOLS=OFF'
    '-DFASTNOISE2_TESTS=OFF'
    '-DFASTNOISE2_UTILITY=OFF'
    "-DCPM_FastSIMD_SOURCE=$(& $fwd $FastSimdDir)"
)
# CMake and FastSIMD write status lines to stderr; PowerShell 5.1 would turn those
# into terminating errors. Judge these calls by their exit code only.
function Invoke-Native {
    param([string] $Exe, [string[]] $Arguments, [string] $Stage)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_ } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "PHASE-FAILED :: $Stage ($LASTEXITCODE)" }
}

Invoke-Native -Exe 'cmake' -Arguments $cmakeArgs -Stage 'Win64 configure'

Write-Host "PHASE :: Win64 build"
Invoke-Native -Exe 'cmake' -Arguments @('--build', (& $fwd $BuildDir)) -Stage 'Win64 build'

$produced = Get-ChildItem -Path $BuildDir -Recurse -Include 'FastNoise.dll', 'FastNoise.lib' -File
if (-not $produced) { throw "PHASE-FAILED :: Win64 build produced no FastNoise.dll/.lib" }
$produced | ForEach-Object { Write-Host ("  {0}  ({1:N0} bytes)" -f $_.FullName, $_.Length) }

Write-Host "PHASE-GREEN :: Win64"
