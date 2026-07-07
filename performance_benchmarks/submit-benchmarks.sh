#!/usr/bin/env bash
set -euo pipefail

STEPS=2000
QUEUE="gpu_4_a100"       # Target A100 GPU nodes (4 GPUs per node)
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
# Question: "How much faster does a fixed-size problem
#            run as you add more networking nodes?"
# Expectation (Amdahl's Law): Runtime decreases with more
# tasks, but speedup saturates when comm_time dominates.
# -------------------------------------------------------
echo ">>> Submitting Strong Scaling Benchmarks (Fixed 8000x8000)..."

STRONG_NX=8000
STRONG_NY=8000

for TASKS in 1 2 4 8 16 32; do
    NODES=$(( (TASKS + 3) / 4 ))
    TASKS_PER_NODE=$(( TASKS / NODES ))
    # Handle case where TASKS < 4 (single node, fewer GPUs)
    if [ "$TASKS" -lt 4 ]; then
        NODES=1
        TASKS_PER_NODE=$TASKS
    fi

    echo "  Strong: tasks=$TASKS, nodes=$NODES, tasks_per_node=$TASKS_PER_NODE"

    # 2D strong scaling
    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Strong2D_${TASKS}" --output="slurm-strong-2d-%j.out" \
           run-bwunicluster.job --dim 2 --Nx $STRONG_NX --Ny $STRONG_NY \
           --steps $STEPS --scaling strong --csv $CSV_FILE

    # 1D strong scaling
    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Strong1D_${TASKS}" --output="slurm-strong-1d-%j.out" \
           run-bwunicluster.job --dim 1 --Nx $STRONG_NX --Ny $STRONG_NY \
           --steps $STEPS --scaling strong --csv $CSV_FILE
done

# -------------------------------------------------------
# WEAK SCALING: Problem size grows with hardware
# Question: "Can the network sustain throughput if the
#            problem size grows proportionally?"
# Expectation (Gustafson's Law): Runtime stays constant
# if network bandwidth scales. Any increase indicates
# MPI synchronization or bandwidth bottleneck.
# -------------------------------------------------------
echo ""
echo ">>> Submitting Weak Scaling Benchmarks (4000x4000 per GPU)..."

BASE_NX_PER_TASK=4000
BASE_NY_PER_TASK=4000

for TASKS in 1 2 4 8 16 32; do
    NODES=$(( (TASKS + 3) / 4 ))
    TASKS_PER_NODE=$(( TASKS / NODES ))
    if [ "$TASKS" -lt 4 ]; then
        NODES=1
        TASKS_PER_NODE=$TASKS
    fi

    # 2D weak scaling: expand domain in both x and y
    # For 2D decomposition, split expansion across both dims
    WEAK_NX=$(( TASKS * BASE_NX_PER_TASK ))
    WEAK_NY=$BASE_NY_PER_TASK

    echo "  Weak 2D: tasks=$TASKS, grid=${WEAK_NX}x${WEAK_NY}"

    sbatch --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE \
           --partition=$QUEUE --gres=gpu:$TASKS_PER_NODE \
           --job-name="Weak2D_${TASKS}" --output="slurm-weak-2d-%j.out" \
           run-bwunicluster.job --dim 2 --Nx $WEAK_NX --Ny $WEAK_NY \
           --steps $STEPS --scaling weak --csv $CSV_FILE

    # 1D weak scaling: expand only in y (1D decomposition is y-slab)
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