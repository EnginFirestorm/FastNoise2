#!/usr/bin/env bash
# Builds libFastNoise.so for Linux x86-64 (native) and Linux arm64 (cross).
# Run from inside WSL, e.g.:
#   wsl.exe -- bash /mnt/d/GIT/FastNoise2Fork/scripts/Build-Linux.sh
#
# The arm64 recipe is the one the fork carried commented out in its CMakeLists;
# it is passed as cache vars here so the source tree stays untouched.
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-/mnt/d/GIT/FastNoise2Fork-v111}"
FASTSIMD_DIR="${FASTSIMD_DIR:-/mnt/d/GIT/FastSIMD-pin}"
CONFIG="${CONFIG:-Release}"
TARGETS="${TARGETS:-x64 arm64}"

echo "PHASE :: Linux toolchain audit"
missing=0
need() {
    if command -v "$1" >/dev/null 2>&1; then
        echo "  ok      $1 -> $(command -v "$1")"
    else
        echo "  MISSING $1"
        missing=1
    fi
}

# Pick the newest installed native g++ instead of pinning one version: the exact
# compiler no longer has to match anything, since this migration does not carry
# old noise output forward. Which one was used is printed, never guessed silently.
pick_native() {
    local v
    for v in 14 13 12 11; do
        if command -v "g++-$v" >/dev/null 2>&1; then
            NATIVE_CXX="g++-$v"; NATIVE_CC="gcc-$v"; return
        fi
    done
    if command -v g++ >/dev/null 2>&1; then
        NATIVE_CXX="g++"; NATIVE_CC="gcc"; return
    fi
    echo "  MISSING g++"
    missing=1
}

need cmake
need ninja
for t in $TARGETS; do
    case "$t" in
        x64)
            pick_native
            [ -n "${NATIVE_CXX:-}" ] && echo "  ok      native  -> $NATIVE_CXX ($($NATIVE_CXX -dumpfullversion))"
            ;;
        arm64) need aarch64-linux-gnu-g++-13 ;;
    esac
done
if [ "$missing" -ne 0 ]; then
    echo "PHASE-FAILED :: Linux toolchain audit"
    echo "  install with e.g.:"
    echo "    sudo apt-get update && sudo apt-get install -y cmake ninja-build g++ g++-13-aarch64-linux-gnu"
    exit 1
fi

[ -f "$SOURCE_DIR/CMakeLists.txt" ]   || { echo "PHASE-FAILED :: no source at $SOURCE_DIR"; exit 1; }
[ -f "$FASTSIMD_DIR/CMakeLists.txt" ] || { echo "PHASE-FAILED :: no FastSIMD at $FASTSIMD_DIR"; exit 1; }

build_one() {
    local tag="$1"; shift
    local build_dir="$SOURCE_DIR/out/build/linux-$tag"
    echo "PHASE :: Linux $tag configure"
    cmake -S "$SOURCE_DIR" -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE="$CONFIG" \
        -DBUILD_SHARED_LIBS=ON \
        -DFASTNOISE2_TOOLS=OFF \
        -DFASTNOISE2_TESTS=OFF \
        -DFASTNOISE2_UTILITY=OFF \
        -DCPM_FastSIMD_SOURCE="$FASTSIMD_DIR" \
        "$@"
    echo "PHASE :: Linux $tag build"
    cmake --build "$build_dir"
    local so
    so=$(find "$build_dir" -name 'libFastNoise.so' -type f | head -1)
    [ -n "$so" ] || { echo "PHASE-FAILED :: Linux $tag produced no libFastNoise.so"; exit 1; }
    echo "  $so ($(stat -c%s "$so") bytes)"
}

for t in $TARGETS; do
    case "$t" in
        x64)
            build_one x64 \
                -DCMAKE_C_COMPILER="$NATIVE_CC" \
                -DCMAKE_CXX_COMPILER="$NATIVE_CXX"
            ;;
        arm64)
            # Clang, not the aarch64 GCC the fork's old recipe named: FastSIMD's
            # NEON.h does not compile with GCC 13 (function-pointer conversion
            # errors in the intrinsic bindings), while Clang builds it clean -
            # which is also what the Android NDK uses for this same code.
            # The GCC cross toolchain still supplies the sysroot.
            clang_cxx=$(command -v clang++-18 || command -v clang++ || true)
            clang_c=$(command -v clang-18 || command -v clang || true)
            if [ -z "$clang_cxx" ]; then
                echo "PHASE-FAILED :: Linux arm64 needs clang (apt install clang-18)"
                exit 1
            fi
            build_one arm64 \
                -DCMAKE_SYSTEM_NAME=Linux \
                -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
                -DCMAKE_C_COMPILER="$clang_c" \
                -DCMAKE_CXX_COMPILER="$clang_cxx" \
                -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu \
                -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu
            ;;
        *) echo "unknown target: $t"; exit 1 ;;
    esac
done

echo "PHASE-GREEN :: Linux"
