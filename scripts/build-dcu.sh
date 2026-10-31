#!/usr/bin/env bash
#
# Optional: configure and build the root project with the Hygon DTK HIP backend.
#
# Plain CPU/CI builds do not need this script. It exists so that a DCU build is
# one reproducible command instead of a hand-assembled set of CMake variables:
# DTK's hip-config.cmake and its try_compile dependency lookups need ROCM_PATH,
# amd_comgr_DIR, AMDDeviceLibs_DIR, hsa-runtime64_DIR and
# CMAKE_TRY_COMPILE_PLATFORM_VARIABLES, which cmake/OneFLOWDTK.cmake now derives
# from a single prefix.
#
# Usage:
#   scripts/build-dcu.sh [--dtk <prefix>] [--arch <gfx906>] [--hip-flags "<flags>"]
#                        [--no-hip-tests] [--jobs N] [--build-dir <dir>]
#                        [--configure-only]
#
# The DTK prefix defaults to $ONEFLOW_DTK_ROOT, then $ROCM_PATH. Everything else
# is optional: without --arch the architecture detection of the target node is
# used, so the same command works on a login node or a compute node.
#
# Run it on a machine that can see the accelerator (a DCU node) when you want
# CMake's device detection to pick the architecture.

set -Eeuo pipefail

# Path-based on purpose: the script must work inside an extracted source
# snapshot (no .git) and with older git builds that lack `git -C`.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "$repo_root"

dtk_root="${ONEFLOW_DTK_ROOT:-${ROCM_PATH:-}}"
arch=""
hip_flags=""
hip_tests="ON"
jobs=""
build_dir="build"
configure_only=false

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dtk) dtk_root="${2:?--dtk needs a path}"; shift 2 ;;
        --arch) arch="${2:?--arch needs a value}"; shift 2 ;;
        --hip-flags) hip_flags="${2:?--hip-flags needs a value}"; shift 2 ;;
        --no-hip-tests) hip_tests="OFF"; shift ;;
        --jobs) jobs="${2:?--jobs needs a number}"; shift 2 ;;
        --build-dir) build_dir="${2:?--build-dir needs a path}"; shift 2 ;;
        --configure-only) configure_only=true; shift ;;
        --help|-h) usage ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -n "$dtk_root" ] || die "no DTK prefix: set ONEFLOW_DTK_ROOT, export ROCM_PATH, or pass --dtk <path>"
[ -d "$dtk_root" ] || die "DTK prefix is not a directory: $dtk_root"
[ -x "$dtk_root/llvm/bin/clang++" ] || die "no HIP compiler under $dtk_root/llvm/bin/clang++"

if [ -z "$jobs" ]; then
    jobs="$( (command -v nproc >/dev/null && nproc) || (command -v sysctl >/dev/null && sysctl -n hw.ncpu) || echo 4 )"
fi

cmake_args=(
    -DONEFLOW_ENABLE_HIP=ON
    -DONEFLOW_ENABLE_HIP_TESTS="$hip_tests"
    -DONEFLOW_DTK_ROOT="$dtk_root"
)

if [ -n "$arch" ]; then
    cmake_args+=( -DONEFLOW_DTK_HIP_ARCHITECTURES="$arch" )
fi

if [ -n "$hip_flags" ]; then
    cmake_args+=( -DONEFLOW_DTK_HIP_FLAGS="$hip_flags" )
fi

printf 'DTK prefix  : %s\n' "$dtk_root"
printf 'build dir   : %s\n' "$build_dir"
printf 'HIP arch    : %s\n' "${arch:-<node detection>}"
printf 'HIP flags   : %s\n' "${hip_flags:-<none>}"
printf 'HIP tests   : %s\n' "$hip_tests"
printf 'cmake       : %s\n' "$(command -v cmake)"
printf 'parallelism : %s\n' "$jobs"

cmake -S "$repo_root" -B "$build_dir" "${cmake_args[@]}"

if [ "$configure_only" = true ]; then
    exit 0
fi

cmake --build "$build_dir" --parallel "$jobs"
