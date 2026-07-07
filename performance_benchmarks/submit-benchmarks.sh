#!/usr/bin/env bash
set -euo pipefail

STEPS=2000
QUEUE="gpu_a100_short"       # Target A100 GPU nodes (4 GPUs per node)
CSV_FILE="benchmark_results.csv"

# Clean previous results
rm -f "$CSV_FILE"
rm -f slurm-strong-*.out slurm-weak-*.out

echo "========================================"
echo "  LBM Benchmark Submission Script"
echo "========================================"
echo "  Steps per run: $STEPS"
echo "  Partition:     $QUEUE"
echo "  CSV output:    $CSV_FILE"
echo "========================================"
echo ""

# -------------------------------------------------------
# STRONG SCALING: Fixed problem size, increasing hardware
# -------------------------------------------------------
echo ">>> Submitting Strong Scaling Benchmarks (Fixed 8000x8000)..."

STRONG_NX=2000
STRONG_NY=4000

for TASKS in 1 2 4 8 16 32; do
    NODES=$(( (TASKS + 3) / 4 ))
    TASKS_PER_NODE=$(( TASKS / NODES ))
    if [ "$TASKS" -lt 4 ]; then
        NODES=1
        TASKS_PER_NODE=$TASKS
    fi

    echo "  Strong: tasks=$TASKS, nodes=$NODES, tasks_per_node=$TASKS_PER_NODE"

    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Strong2D_${TASKS}" --output="slurm-strong-2d-%j.out" \
           run-bwunicluster.job --dim 2 --Nx $STRONG_NX --Ny $STRONG_NY \
           --steps $STEPS --scaling strong --csv $CSV_FILE

    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Strong1D_${TASKS}" --output="slurm-strong-1d-%j.out" \
           run-bwunicluster.job --dim 1 --Nx $STRONG_NX --Ny $STRONG_NY \
           --steps $STEPS --scaling strong --csv $CSV_FILE
done

# -------------------------------------------------------
# WEAK SCALING: Problem size grows with hardware
# 2D decomposition grows BOTH dimensions in alternation
# (quadratic/balanced growth): every two doublings of TASKS,
# both Nx and Ny have doubled, so area = TASKS * base_area.
#
#   TASKS=1  -> 4k x 4k
#   TASKS=2  -> 4k x 8k
#   TASKS=4  -> 8k x 8k
#   TASKS=8  -> 8k x 16k
#   TASKS=16 -> 16k x 16k
#   TASKS=32 -> 16k x 32k
# -------------------------------------------------------
echo ""
echo ">>> Submitting Weak Scaling Benchmarks (balanced 2D growth)..."

BASE_NX_PER_TASK=2000
BASE_NY_PER_TASK=2000

for TASKS in 1 2 4 8 16 32; do
    NODES=$(( (TASKS + 3) / 4 ))
    TASKS_PER_NODE=$(( TASKS / NODES ))
    if [ "$TASKS" -lt 4 ]; then
        NODES=1
        TASKS_PER_NODE=$TASKS
    fi

    # ---- 2D weak scaling: balanced (alternating) growth ----
    # Compute log2(TASKS)
    doubles=0
    n=$TASKS
    while [ "$n" -gt 1 ]; do
        n=$(( n / 2 ))
        doubles=$(( doubles + 1 ))
    done

    half=$(( doubles / 2 ))
    if [ $(( doubles % 2 )) -eq 0 ]; then
        factor_x=$(( 2 ** half ))
        factor_y=$factor_x
    else
        factor_x=$(( 2 ** half ))
        factor_y=$(( 2 ** (half + 1) ))
    fi

    WEAK_NX=$(( BASE_NX_PER_TASK * factor_x ))
    WEAK_NY=$(( BASE_NY_PER_TASK * factor_y ))

    echo "  Weak 2D: tasks=$TASKS, grid=${WEAK_NX}x${WEAK_NY} (factors ${factor_x}x${factor_y})"

    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Weak2D_${TASKS}" --output="slurm-weak-2d-%j.out" \
           run-bwunicluster.job --dim 2 --Nx $WEAK_NX --Ny $WEAK_NY \
           --steps $STEPS --scaling weak --csv $CSV_FILE

    # ---- 1D weak scaling: expand only in y (y-slab decomposition) ----
    WEAK_NY_1D=$(( TASKS * BASE_NY_PER_TASK ))

    echo "  Weak 1D: tasks=$TASKS, grid=${BASE_NX_PER_TASK}x${WEAK_NY_1D}"

    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Weak1D_${TASKS}" --output="slurm-weak-1d-%j.out" \
           run-bwunicluster.job --dim 1 --Nx $BASE_NX_PER_TASK --Ny $WEAK_NY_1D \
           --steps $STEPS --scaling weak --csv $CSV_FILE
done

echo ""
echo "========================================"
echo "  All jobs submitted!"
echo "  Check status:  squeue -u \$USER"
echo "  CSV results:   $CSV_FILE"
echo "========================================"