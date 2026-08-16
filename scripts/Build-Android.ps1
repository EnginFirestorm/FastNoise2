<#
.SYNOPSIS
    Builds libFastNoise.so + libFastNoise.a for Android arm64-v8a and x86_64.

.DESCRIPTION
    The historical build used NDK r23c, which is no longer installed on this
    machine. The NDK is therefore probed: explicit -NdkRoot, then the old r23c
    location, then $env:NDKROOT, then the newest NDK under the Android SDK.
    Because this migration abandons old noise output anyway, a newer NDK is fine.

    Both a shared and a static library are produced per ABI: Build.cs links the
    .a, Android.xml packages the .so.
#>
[CmdletBinding()]
param(
    [string]   $SourceDir   = 'D:\GIT\FastNoise2Fork-v111',
    [string]   $FastSimdDir = 'D:\GIT\FastSIMD-pin',
    [string]   $NdkRoot,
    [string]   $NinjaPath,
    [string[]] $Abis        = @('arm64-v8a', 'x86_64'),
    [string]   $Platform    = 'android-26',
    [string]   $Config      = 'Release'
)

$ErrorActionPreference = 'Stop'
$fwd = { param($p) $p -replace '\\', '/' }

function Invoke-Native {
    param([string] $Exe, [string[]] $Arguments, [string] $Stage)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host $_ } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "PHASE-FAILED :: $Stage ($LASTEXITCODE)" }
}

# --- NDK ---------------------------------------------------------------------
if (-not $NdkRoot) {
    $candidates = @('C:\Microsoft\AndroidNDK\android-ndk-r23c')
    if ($env:NDKROOT) { $candidates += $env:NDKROOT }
    $sdkNdk = Join-Path $env:LOCALAPPDATA 'Android\Sdk\ndk'
    if (Test-Path $sdkNdk) {
        $candidates += (Get-ChildItem $sdkNdk -Directory |
            Sort-Object { [version]($_.Name -replace '[^0-9.].*$', '') } -Descending |
            Select-Object -ExpandProperty FullName)
    }
    foreach ($c in $candidates) {
        if ($c -and (Test-Path (Join-Path $c 'build\cmake\android.toolchain.cmake'))) { $NdkRoot = $c; break }
    }
}
if (-not $NdkRoot) { throw "No Android NDK found; pass -NdkRoot." }
if ($NdkRoot -notmatch 'r23c') {
    Write-Host "NOTE: building with $NdkRoot (the original shipped libs used NDK r23c)."
}

if (-not $NinjaPath) {
    $vsNinja = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio', 'D:\Program Files\Microsoft Visual Studio' `
        -Recurse -Filter 'ninja.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vsNinja) { $NinjaPath = $vsNinja.FullName }
    else {
        $onPath = Get-Command ninja -ErrorAction SilentlyContinue
        if ($onPath) { $NinjaPath = $onPath.Source } else { throw "ninja.exe not found; pass -NinjaPath." }
    }
}

$toolchain = Join-Path $NdkRoot 'build\cmake\android.toolchain.cmake'
Write-Host "PHASE :: Android"
Write-Host "  NDK      : $NdkRoot"
Write-Host "  ninja    : $NinjaPath"
Write-Host "  ABIs     : $($Abis -join ', ')"

foreach ($abi in $Abis) {
    foreach ($shared in @('ON', 'OFF')) {
        $tag = if ($shared -eq 'ON') { 'shared' } else { 'static' }
        $buildDir = Join-Path $SourceDir "out\build\android-$abi-$tag"
        Write-Host "PHASE :: Android $abi ($tag)"

        $args = @(
            '-S', (& $fwd $SourceDir)
            '-B', (& $fwd $buildDir)
            '-G', 'Ninja'
            "-DCMAKE_MAKE_PROGRAM=$(& $fwd $NinjaPath)"
            "-DCMAKE_TOOLCHAIN_FILE=$(& $fwd $toolchain)"
            "-DANDROID_ABI=$abi"
            "-DANDROID_PLATFORM=$Platform"
            "-DCMAKE_BUILD_TYPE=$Config"
            "-DBUILD_SHARED_LIBS=$shared"
            '-DFASTNOISE2_TOOLS=OFF'
            '-DFASTNOISE2_TESTS=OFF'
            '-DFASTNOISE2_UTILITY=OFF'
            "-DCPM_FastSIMD_SOURCE=$(& $fwd $FastSimdDir)"
        )
        if ($abi -eq 'arm64-v8a') { $args += '-DANDROID_ARM_MODE=Arm' }

        Invoke-Native -Exe 'cmake' -Arguments $args -Stage "Android $abi $tag configure"
        Invoke-Native -Exe 'cmake' -Arguments @('--build', (& $fwd $buildDir)) -Stage "Android $abi $tag build"

        $wanted = if ($shared -eq 'ON') { 'libFastNoise.so' } else { 'libFastNoise.a' }
        $produced = Get-ChildItem -Path $buildDir -Recurse -Filter $wanted -File
        if (-not $produced) { throw "PHASE-FAILED :: Android $abi $tag produced no $wanted" }
        $produced | ForEach-Object { Write-Host ("  {0}  ({1:N0} bytes)" -f $_.FullName, $_.Length) }
    }
}

Write-Host "PHASE-GREEN :: Android"
