<#
.SYNOPSIS
    Copies built FastNoise2 binaries and the consumer header tree into the
    PorismDIMsWorldGenerator plugin.

.DESCRIPTION
    Header set mirrors the library's own install rules (src/CMakeLists.txt):
      include/FastNoise/**/*.h
      FastSIMD/DispatchClass.h
      FastSIMD/Utility/*.h
    The .inl files are implementation-only (compiled inside the library) and are
    deliberately NOT vendored. The generated FastSIMD_FastNoise_config.h is also
    skipped: no public header reaches it, and it is per-platform, so vendoring it
    would force a per-platform include dir for no gain.

    Run with -WhatIf to preview.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string] $SourceDir   = 'D:\GIT\FastNoise2Fork-v111',
    [string] $FastSimdDir = 'D:\GIT\FastSIMD-pin',
    [string] $PluginThirdParty = 'D:\GIT\PorismDIMsWorldGen\Plugins\PorismDIMsWorldGenerator\Source\ThirdParty\FastNoise2',
    [switch] $SkipHeaders,
    [switch] $SkipLibs,
    # Keep the debug information in the copies. Only for chasing a crash inside
    # FastNoise itself — it multiplies the vendored size by roughly five, and
    # those bytes end up in git history and in the customer's package.
    [switch] $NoStrip
)

$ErrorActionPreference = 'Stop'

# llvm-strip handles every ELF architecture, so one tool covers Android and both
# Linux targets. Located once; a missing NDK is a hard error rather than a silent
# 125 MB of debug data going into the repo and the customer package.
function Get-LlvmStrip {
    if ($script:LlvmStripPath) { return $script:LlvmStripPath }

    $roots = @()
    if ($env:NDKROOT) { $roots += $env:NDKROOT }
    $sdkNdk = Join-Path $env:LOCALAPPDATA 'Android\Sdk\ndk'
    if (Test-Path $sdkNdk) {
        $roots += (Get-ChildItem $sdkNdk -Directory | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
    }
    foreach ($r in $roots) {
        $candidate = Join-Path $r 'toolchains\llvm\prebuilt\windows-x86_64\bin\llvm-strip.exe'
        if (Test-Path $candidate) { $script:LlvmStripPath = $candidate; return $candidate }
    }
    throw ("llvm-strip.exe not found (looked at NDKROOT and $sdkNdk). It is required: an unstripped " +
           "Release build is ~125 MB instead of ~27 MB, because the NDK adds -g to every configuration.")
}

# What may be removed differs by artifact kind, and getting it wrong is silent:
#   .so  --strip-unneeded : drops debug + local symbols, KEEPS .dynsym and SONAME,
#                           which is all the dynamic loader and Android's
#                           soLoadLibrary ever consult.
#   .a   --strip-debug    : drops debug sections ONLY. The symbol table must stay,
#                           or the Android link step cannot resolve anything.
# Verified 2026-08-03 on all six ELF artifacts: defined-symbol and dynamic-symbol
# counts were byte-for-byte identical before and after.
function Invoke-StripArtifact {
    param([string] $Path)

    $ext = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
    if ($ext -ne '.so' -and $ext -ne '.a') { return }   # MSVC keeps debug info in a .pdb we never copy

    $strip = Get-LlvmStrip
    $flag  = if ($ext -eq '.a') { '--strip-debug' } else { '--strip-unneeded' }
    $before = (Get-Item $Path).Length

    & $strip $flag $Path
    if ($LASTEXITCODE -ne 0) { throw "llvm-strip $flag failed on '$Path' (exit $LASTEXITCODE)." }

    $after = (Get-Item $Path).Length
    Write-Host ("    stripped {0} {1:N0} -> {2:N0} bytes" -f $flag, $before, $after)
}

function Copy-Artifact {
    param([string] $From, [string] $To)
    if (-not (Test-Path $From)) { throw "Missing build artifact: $From" }
    $oldHash = if (Test-Path $To) { (Get-FileHash $To -Algorithm SHA256).Hash.Substring(0, 12) } else { '(new)' }
    $dir = Split-Path $To -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if ($PSCmdlet.ShouldProcess($To, 'copy')) {
        Copy-Item -Path $From -Destination $To -Force
        # Strip the COPY, never the build output: a rebuild would otherwise have
        # to run again to get debuggable binaries back.
        if (-not $NoStrip) { Invoke-StripArtifact -Path $To }
        $newHash = (Get-FileHash $To -Algorithm SHA256).Hash.Substring(0, 12)
        Write-Host ("  {0,-42} {1} -> {2}  ({3:N0} bytes)" -f (Split-Path $To -Leaf), $oldHash, $newHash, (Get-Item $To).Length)
    }
}

# --- libraries ---------------------------------------------------------------
if (-not $SkipLibs) {
    Write-Host "PHASE :: copy libs"
    $map = @(
        @{ From = "$SourceDir\out\build\win64\Release\bin\FastNoise.dll";                   To = "$PluginThirdParty\libs\Windows-x86-64\FastNoise.dll" }
        @{ From = "$SourceDir\out\build\win64\Release\lib\FastNoise.lib";                   To = "$PluginThirdParty\libs\Windows-x86-64\FastNoise.lib" }
        @{ From = "$SourceDir\out\build\android-arm64-v8a-shared\Release\lib\libFastNoise.so"; To = "$PluginThirdParty\libs\Android-arm64-v8a\libFastNoise.so" }
        @{ From = "$SourceDir\out\build\android-arm64-v8a-static\Release\lib\libFastNoise.a";  To = "$PluginThirdParty\libs\Android-arm64-v8a\libFastNoise.a" }
        @{ From = "$SourceDir\out\build\android-x86_64-shared\Release\lib\libFastNoise.so";    To = "$PluginThirdParty\libs\Android-x86_64\libFastNoise.so" }
        @{ From = "$SourceDir\out\build\android-x86_64-static\Release\lib\libFastNoise.a";     To = "$PluginThirdParty\libs\Android-x86_64\libFastNoise.a" }
        @{ From = "$SourceDir\out\build\linux-x64\Release\lib\libFastNoise.so";             To = "$PluginThirdParty\libs\Linux-x86-64\libFastNoise.so" }
        @{ From = "$SourceDir\out\build\linux-arm64\Release\lib\libFastNoise.so";           To = "$PluginThirdParty\libs\Linux-arm64\libFastNoise.so" }
    )
    foreach ($m in $map) {
        if (Test-Path $m.From) { Copy-Artifact -From $m.From -To $m.To }
        else { Write-Warning "not built yet, keeping existing: $($m.To)" }
    }
}

# --- headers -----------------------------------------------------------------
if (-not $SkipHeaders) {
    Write-Host "PHASE :: sync headers"
    $incDst = Join-Path $PluginThirdParty 'include'
    if ((Test-Path $incDst) -and $PSCmdlet.ShouldProcess($incDst, 'replace header tree')) {
        Remove-Item -Recurse -Force $incDst
    }
    if ($PSCmdlet.ShouldProcess($incDst, 'create header tree')) {
        New-Item -ItemType Directory -Force -Path $incDst | Out-Null

        $fnSrc = Join-Path $SourceDir 'include\FastNoise'
        Get-ChildItem $fnSrc -Recurse -Filter '*.h' -File | ForEach-Object {
            $rel = $_.FullName.Substring($fnSrc.Length).TrimStart('\')
            $dst = Join-Path (Join-Path $incDst 'FastNoise') $rel
            New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
            Copy-Item $_.FullName $dst -Force
        }

        $fsUtil = Join-Path $FastSimdDir 'include\FastSIMD\Utility'
        $fsDst = Join-Path $incDst 'FastSIMD'
        New-Item -ItemType Directory -Force -Path (Join-Path $fsDst 'Utility') | Out-Null
        Get-ChildItem $fsUtil -Filter '*.h' -File | ForEach-Object {
            Copy-Item $_.FullName (Join-Path (Join-Path $fsDst 'Utility') $_.Name) -Force
        }
        Copy-Item (Join-Path $FastSimdDir 'include\FastSIMD\DispatchClass.h') (Join-Path $fsDst 'DispatchClass.h') -Force

        $count = (Get-ChildItem $incDst -Recurse -File).Count
        Write-Host "  vendored $count headers into $incDst"
    }
}

Write-Host "PHASE-GREEN :: copy"
