#!/bin/bash -l
# Compare one-step main-solver traces for legacy CPU, CPU batch, and HIP batch.
#SBATCH --job-name=oneflow_f2_stage
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=27G
#SBATCH --gres=dcu:1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${ONEFLOW_F2_STAGE_SOURCE_DIR:?Set ONEFLOW_F2_STAGE_SOURCE_DIR}"
: "${ONEFLOW_F2_STAGE_BINARY:?Set ONEFLOW_F2_STAGE_BINARY}"

module purge
module load compiler/dtk/26.04
module load mpi/openmpi/gcc-9.3.0/4.1.5
module load python/3.8.10

source_case="$ONEFLOW_F2_STAGE_SOURCE_DIR/test/plateuns2dslau2"
trace_root="${ONEFLOW_F2_STAGE_RUN_DIR:-$PWD/f2-main-solver-stage-${SLURM_JOB_ID:-manual}}"

if [ -e "$trace_root" ]; then
    echo "Refusing to overwrite existing F2 stage run directory" >&2
    exit 2
fi
mkdir -p "$trace_root"
for mode in legacy cpu_batch hip_batch; do
    cp -a "$source_case" "$trace_root/$mode"
    rm -rf "$trace_root/$mode/results" "$trace_root/$mode/restart" \
        "$trace_root/$mode/log"
    mkdir -p "$trace_root/$mode/results" "$trace_root/$mode/restart" \
        "$trace_root/$mode/log"
    sed -i -E 's/^int[[:space:]]+maxSteps[[:space:]]*=.*/int maxSteps = 1;/' \
        "$trace_root/$mode/script/cfd.txt"
    sed -i -E 's/^int[[:space:]]+ischeme[[:space:]]*=.*/int ischeme = 5; \/\/ Lax-Friedrichs F2 stage gate/' \
        "$trace_root/$mode/script/cfd.txt"
done

binary_dir=$(dirname "$ONEFLOW_F2_STAGE_BINARY")
if [ ! -d "$binary_dir/system" ]; then
    cp -a "$ONEFLOW_F2_STAGE_SOURCE_DIR/system" "$binary_dir/system"
fi

run_case() {
    local mode=$1
    rm -f "$trace_root/$mode.stage"
    (
        cd "$trace_root"
        export ONEFLOW_UNS_STAGE_TRACE_FILE="$trace_root/$mode.stage"
        unset ONEFLOW_ENABLE_UNS_CPU_BATCH ONEFLOW_ENABLE_UNS_HIP_BATCH || true
        if [ "$mode" = hip_batch ]; then
            export ONEFLOW_ACCEL_BACKEND=HIP
            export ONEFLOW_ENABLE_UNS_HIP_BATCH=1
        else
            export ONEFLOW_ACCEL_BACKEND=CPU
            if [ "$mode" = cpu_batch ]; then
                export ONEFLOW_ENABLE_UNS_CPU_BATCH=1
            fi
        fi
        timeout --signal=TERM --kill-after=10s 180 \
            mpirun -np 1 "$ONEFLOW_F2_STAGE_BINARY" 0 "$mode"
    ) > "$trace_root/$mode.log" 2>&1
}

run_case legacy
run_case cpu_batch
run_case hip_batch

python3 "$ONEFLOW_F2_STAGE_SOURCE_DIR/ci/kunshan/f2-main-solver-stage-verify.py" \
    "$trace_root/legacy.stage" \
    "$trace_root/cpu_batch.stage" \
    "$trace_root/hip_batch.stage" | tee "$trace_root/verify.txt"
