#!/bin/bash -l
# Run the 3D main solver for a longer stability gate without writing the
# multi-gigabyte full-array trace used by the short accuracy gate.
#SBATCH --job-name=oneflow_f2_stability
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=27G
#SBATCH --gres=dcu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${ONEFLOW_F2_STABILITY_SOURCE_DIR:?Set ONEFLOW_F2_STABILITY_SOURCE_DIR}"
: "${ONEFLOW_F2_STABILITY_BINARY:?Set ONEFLOW_F2_STABILITY_BINARY}"
: "${ONEFLOW_F2_STABILITY_CASE:?Set ONEFLOW_F2_STABILITY_CASE (relative to source)}"

module purge
module load compiler/gcc/9.3.0
module load mpi/openmpi/gcc-9.3.0/4.1.5
module load compiler/dtk/26.04

clang_bin="$(command -v clang)"
llvm_root="$(dirname "$(dirname "$clang_bin")")"
dtk_root="$(dirname "$llvm_root")"
mpi_root="/public/software/mpi/openmpi/openmpi-4.1.5"
metis_root="$HOME/.scnet-hpc/oneflow/deps/metis-install"
export PATH="$dtk_root/bin:$llvm_root/bin:$PATH"
export LD_LIBRARY_PATH="$dtk_root/hip/lib:$llvm_root/lib:$dtk_root/lib:$dtk_root/lib64:$dtk_root/dcc/lib64:$dtk_root/dcc/comgr/lib64:$metis_root/lib:$mpi_root/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

steps="${ONEFLOW_F2_STABILITY_STEPS:-50}"
timeout_seconds="${ONEFLOW_F2_STABILITY_TIMEOUT_SECONDS:-900}"
cpu_ranks="${ONEFLOW_F2_STABILITY_CPU_RANKS:-8}"
hip_ranks="${ONEFLOW_F2_STABILITY_HIP_RANKS:-1}"
time_integral="${ONEFLOW_F2_STABILITY_TIME_INTEGRAL:-1}"
rk_stages="${ONEFLOW_F2_STABILITY_RK_STAGES:-3}"
cfl_start="${ONEFLOW_F2_STABILITY_CFL:-0.01}"
cfl_end="${ONEFLOW_F2_STABILITY_CFL_END:-$cfl_start}"
disable_viscous="${ONEFLOW_F2_STABILITY_DISABLE_VISCOUS:-1}"
disable_limiter="${ONEFLOW_F2_STABILITY_DISABLE_LIMITER:-1}"
limiter="${ONEFLOW_F2_STABILITY_LIMITER:-}"
stability_root="${ONEFLOW_F2_STABILITY_RUN_DIR:-$PWD/f2-main-solver-stability-${SLURM_JOB_ID:-manual}}"

for value_name in steps timeout_seconds cpu_ranks hip_ranks time_integral rk_stages; do
    value="${!value_name}"
    case "$value" in
        ''|*[!0-9]*|0) echo "Invalid $value_name=$value" >&2; exit 2 ;;
    esac
done
for value_name in cfl_start cfl_end; do
    value="${!value_name}"
    if ! awk -v value="$value" \
        'BEGIN {
            valid = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/;
            exit !(valid && value + 0.0 > 0.0);
        }'
    then
        echo "Invalid $value_name=$value" >&2
        exit 2
    fi
done

for value_name in disable_viscous disable_limiter; do
    value="${!value_name}"
    case "$value" in
        0|1) ;;
        *) echo "Invalid $value_name=$value (expected 0 or 1)" >&2; exit 2 ;;
    esac
done

if [ -n "$limiter" ]; then
    case "$limiter" in
        -1|0|1|2) ;;
        *) echo "Invalid limiter=$limiter (expected -1, 0, 1 or 2)" >&2; exit 2 ;;
    esac
elif [ "$disable_limiter" = 1 ]; then
    limiter=0
fi

if [ -e "$stability_root" ]; then
    echo "Refusing to overwrite existing F2 stability run directory" >&2
    exit 2
fi
mkdir -p "$stability_root"

source_case="$ONEFLOW_F2_STABILITY_SOURCE_DIR/$ONEFLOW_F2_STABILITY_CASE"
for mode in legacy cpu_batch hip_batch; do
    cp -a "$source_case" "$stability_root/$mode"
    rm -rf "$stability_root/$mode/results" "$stability_root/$mode/restart" \
        "$stability_root/$mode/log"
    mkdir -p "$stability_root/$mode/results" "$stability_root/$mode/restart" \
        "$stability_root/$mode/log"
    sed -i -E "s/^int[[:space:]]+maxSteps[[:space:]]*=.*/int maxSteps = ${steps};/" \
        "$stability_root/$mode/script/cfd.txt"
    sed -i -E 's/^int[[:space:]]+ischeme[[:space:]]*=.*/int ischeme = 5; \/\/ Lax-Friedrichs F stability gate/' \
        "$stability_root/$mode/script/cfd.txt"
    sed -i -E "s/^int[[:space:]]+time_integral[[:space:]]*=.*/int time_integral = ${time_integral};/" \
        "$stability_root/$mode/script/cfd.txt"
    sed -i -E "s/^int[[:space:]]+rk_stage[[:space:]]*=.*/int rk_stage = ${rk_stages};/" \
        "$stability_root/$mode/script/cfd.txt"
    sed -i -E "s/^Real[[:space:]]+cflst[[:space:]]*=.*/Real cflst = ${cfl_start};/" \
        "$stability_root/$mode/script/cfd.txt"
    sed -i -E "s/^Real[[:space:]]+cfled[[:space:]]*=.*/Real cfled = ${cfl_end};/" \
        "$stability_root/$mode/script/cfd.txt"
    if [ "$disable_viscous" = 1 ]; then
        sed -i -E 's/^int[[:space:]]+vismodel[[:space:]]*=.*/int vismodel = 0; \/\/ F stability inviscid gate/' \
            "$stability_root/$mode/script/vismodel.txt"
        sed -i -E '/^[[:space:]]*TurbSolver[[:space:]]*$/d' \
            "$stability_root/$mode/script/solver.txt"
    fi
    if [ -n "$limiter" ]; then
        sed -i -E "s/^int[[:space:]]+ilim[[:space:]]*=.*/int ilim = ${limiter}; \/\/ F stability limiter gate/" \
            "$stability_root/$mode/script/cfd.txt"
    fi
done

binary_dir=$(dirname "$ONEFLOW_F2_STABILITY_BINARY")
if [ ! -d "$binary_dir/system" ]; then
    cp -a "$ONEFLOW_F2_STABILITY_SOURCE_DIR/system" "$binary_dir/system"
fi

summary_file="$stability_root/stability.tsv"
printf 'mode\tranks\texit_code\tdiagnostic_hits\tstatus\n' > "$summary_file"
config_file="$stability_root/configuration.tsv"
{
    printf 'key\tvalue\n'
    printf 'steps\t%s\n' "$steps"
    printf 'time_integral\t%s\n' "$time_integral"
    printf 'rk_stages\t%s\n' "$rk_stages"
    printf 'cfl_start\t%s\n' "$cfl_start"
    printf 'cfl_end\t%s\n' "$cfl_end"
    printf 'disable_viscous\t%s\n' "$disable_viscous"
    printf 'limiter\t%s\n' "${limiter:-source}"
    printf 'cpu_ranks\t%s\n' "$cpu_ranks"
    printf 'hip_ranks\t%s\n' "$hip_ranks"
} > "$config_file"
overall=0

run_mode() {
    local mode=$1
    local ranks=$2
    local log_file="$stability_root/$mode.log"
    local rc diagnostic_hits status

    set +e
    (
        cd "$stability_root"
        unset ONEFLOW_ENABLE_UNS_CPU_BATCH ONEFLOW_ENABLE_UNS_HIP_BATCH \
            ONEFLOW_UNS_STAGE_TRACE_FILE ONEFLOW_STAGE_BREAKDOWN_FILE || true
        app_prefix=()
        if [ "${ONEFLOW_F2_STABILITY_FAIL_ON_NONPHYSICAL:-1}" = 1 ]; then
            app_prefix=( env ONEFLOW_FAIL_ON_NONPHYSICAL=1 )
        fi
        if [ "$mode" = hip_batch ]; then
            export ONEFLOW_ACCEL_BACKEND=HIP
            export ONEFLOW_ENABLE_UNS_HIP_BATCH=1
        else
            export ONEFLOW_ACCEL_BACKEND=CPU
            if [ "$mode" = cpu_batch ]; then
                export ONEFLOW_ENABLE_UNS_CPU_BATCH=1
            fi
        fi
        timeout --foreground --signal=TERM --kill-after=10s "$timeout_seconds" \
            mpirun -np "$ranks" "${app_prefix[@]}" "$ONEFLOW_F2_STABILITY_BINARY" 0 "$mode"
    ) > "$log_file" 2>&1
    rc=$?
    set -e

    diagnostic_hits=$(grep -Eic \
        '(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)|non.?physical|negative[[:space:]]+(density|pressure)|(density|pressure)[^[:cntrl:]]*<=' \
        "$log_file" || true)
    status=PASS
    if [ "$rc" -ne 0 ] || [ "$diagnostic_hits" -ne 0 ]; then
        status=FAIL
        overall=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$ranks" "$rc" "$diagnostic_hits" "$status" | tee -a "$summary_file"
}

# Preserve the diagnostic order: establish legacy CPU stability first, then
# compare CPU batch, and only then run the HIP batch path.
run_mode legacy "$cpu_ranks"
run_mode cpu_batch "$cpu_ranks"
run_mode hip_batch "$hip_ranks"

cat "$summary_file"
exit "$overall"
