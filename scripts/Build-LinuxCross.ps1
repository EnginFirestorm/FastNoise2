<#
.SYNOPSIS
    Builds libFastNoise.so for Linux x86-64 and Linux arm64 on Windows, with Unreal's own
    Linux cross toolchain - no WSL needed.

.DESCRIPTION
    The toolchain is the one Unreal packages its Linux targets with (LINUX_MULTIARCH_ROOT,
    e.g. C:\UnrealToolchains\v26_clang-20.1.8-rockylinux8\): one clang for every target plus a
    sysroot per target (x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnueabi). The library
    links the sysroot's libstdc++ dynamically, as the WSL build (Build-Linux.sh) did.

    cmake and ninja are Visual Studio's (found with vswhere), as in Build-Win64.ps1.

    Outputs: out\build\linux-x64 and out\build\linux-arm64 - the folders Copy-LibsToPlugin.ps1
    reads.
#>
[CmdletBinding()]
param(
    [string] $SourceDir,
    [string] $FastSimdDir = 'D:\GIT\FastSIMD-pin',
    [string] $ToolchainRoot = $env:LINUX_MULTIARCH_ROOT,
    [string] $Config = 'Release',
    [string[]] $Targets = @('x64', 'arm64')
)

$ErrorActionPreference = 'Stop'
if (-not $SourceDir) { $SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent }

if (-not $ToolchainRoot -or -not (Test-Path $ToolchainRoot)) { throw "No Linux toolchain: set LINUX_MULTIARCH_ROOT or pass -ToolchainRoot." }
if (-not (Test-Path (Join-Path $FastSimdDir 'CMakeLists.txt'))) { throw "No FastSIMD source in $FastSimdDir" }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found" }
$vs = & $vswhere -latest -products * -property installationPath
$cmake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ninja = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
foreach ($p in @($cmake, $ninja)) { if (-not (Test-Path $p)) { throw "not found: $p" } }

$bin = Join-Path $ToolchainRoot 'x86_64-unknown-linux-gnu\bin'
$clang = Join-Path $bin 'clang.exe'
$clangxx = Join-Path $bin 'clang++.exe'
$ar = Join-Path $bin 'llvm-ar.exe'
$ranlib = Join-Path $bin 'llvm-ranlib.exe'
foreach ($p in @($clang, $clangxx, $ar, $ranlib)) { if (-not (Test-Path $p)) { throw "not found: $p" } }

$fwd = { param($p) $p -replace '\\', '/' }

function Invoke-Native {
    param([string] $Exe, [string[]] $Arguments, [string] $Stage)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_ } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "PHASE-FAILED :: $Stage ($LASTEXITCODE)" }
}

$triples = @{ x64 = @('x86_64-unknown-linux-gnu', 'x86_64'); arm64 = @('aarch64-unknown-linux-gnueabi', 'aarch64') }

foreach ($t in $Targets) {
    if (-not $triples.ContainsKey($t)) { throw "unknown target: $t" }
    $triple, $proc = $triples[$t]
    $sysroot = Join-Path $ToolchainRoot $triple
    if (-not (Test-Path $sysroot)) { throw "no sysroot for $t at $sysroot" }
    $buildDir = Join-Path $SourceDir "out\build\linux-$t"

    Write-Host "PHASE :: Linux $t configure ($triple, sysroot $sysroot)"
    $cmakeArgs = @(
        '-S', (& $fwd $SourceDir)
        '-B', (& $fwd $buildDir)
        '-G', 'Ninja'
        "-DCMAKE_MAKE_PROGRAM=$(& $fwd $ninja)"
        "-DCMAKE_BUILD_TYPE=$Config"
        '-DCMAKE_SYSTEM_NAME=Linux'
        "-DCMAKE_SYSTEM_PROCESSOR=$proc"
        "-DCMAKE_C_COMPILER=$(& $fwd $clang)"
        "-DCMAKE_CXX_COMPILER=$(& $fwd $clangxx)"
        "-DCMAKE_C_COMPILER_TARGET=$triple"
        "-DCMAKE_CXX_COMPILER_TARGET=$triple"
        "-DCMAKE_SYSROOT=$(& $fwd $sysroot)"
        "-DCMAKE_AR=$(& $fwd $ar)"
        "-DCMAKE_RANLIB=$(& $fwd $ranlib)"
        '-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld'
        '-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld'
        '-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER'
        '-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY'
        '-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY'
        '-DBUILD_SHARED_LIBS=ON'
        '-DFASTNOISE2_TOOLS=OFF'
        '-DFASTNOISE2_TESTS=OFF'
        '-DFASTNOISE2_UTILITY=OFF'
        "-DCPM_FastSIMD_SOURCE=$(& $fwd $FastSimdDir)"
    )
    Invoke-Native -Exe $cmake -Arguments $cmakeArgs -Stage "Linux $t configure"
    Write-Host "PHASE :: Linux $t build"
    Invoke-Native -Exe $cmake -Arguments @('--build', (& $fwd $buildDir)) -Stage "Linux $t build"
    $so = Get-ChildItem -Path $buildDir -Recurse -Filter 'libFastNoise.so' -File | Select-Object -First 1
    if (-not $so) { throw "PHASE-FAILED :: Linux $t produced no libFastNoise.so" }
    Write-Host ("  {0}  ({1:N0} bytes)" -f $so.FullName, $so.Length)
}

Write-Host "PHASE-GREEN :: Linux (cross)"
