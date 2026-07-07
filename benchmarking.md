# Benchmarking my implementation

I decided to use seperate timers for networking communication and the calculation of the fluid simulation.

I decided to move forward with 2 benchmarks that allow me to find out about the follwoing things:

1. How much faster does a fixed-size problem run as you add more networking nodes?
2. Can the network sustain throughput if the problem size grows proportionally with the hardware?

The first question (describing Amdahl's Law) is decided by using a large grid, and adding more and more nodes to it. e.g. grid: 10k x 10k; nodes: 1,2,4,8,16; Ranks fixed to e.g. 40.

The second question is answered by keeping the work per rank constant.
* 1 Node with 40 ranks -> global grid 2k x 4x
* 2 Nodes with 80 ranks -> global grid 2k x 8k
* 4 Nodes with 160 ranks -> global grid 2k x 16k

This is done first to check the implementation of the 1d comain decomposition.

I then imple



```text
[fr_tw313@uc3n991 performance_benchmarks]$ ./submit-benchmarks.sh
========================================
  LBM Benchmark Submission Script
========================================
  Steps per run: 2000
  Partition:     gpu_a100_short
  CSV output:    benchmark_results.csv
========================================

>>> Submitting Strong Scaling Benchmarks (Fixed 8000x8000)...
  Strong: tasks=1, nodes=1, tasks_per_node=1
Submitted batch job 5833312
Submitted batch job 5833313
  Strong: tasks=2, nodes=1, tasks_per_node=2
Submitted batch job 5833314
Submitted batch job 5833315
  Strong: tasks=4, nodes=1, tasks_per_node=4
Submitted batch job 5833316
Submitted batch job 5833317
  Strong: tasks=8, nodes=2, tasks_per_node=4
Submitted batch job 5833318
Submitted batch job 5833319
  Strong: tasks=16, nodes=4, tasks_per_node=4
Submitted batch job 5833320
Submitted batch job 5833321
  Strong: tasks=32, nodes=8, tasks_per_node=4
Submitted batch job 5833322
Submitted batch job 5833323

>>> Submitting Weak Scaling Benchmarks (balanced 2D growth)...
  Weak 2D: tasks=1, grid=2000x2000 (factors 1x1)
Submitted batch job 5833324
  Weak 1D: tasks=1, grid=2000x2000
Submitted batch job 5833325
  Weak 2D: tasks=2, grid=2000x4000 (factors 1x2)
Submitted batch job 5833326
  Weak 1D: tasks=2, grid=2000x4000
Submitted batch job 5833327
  Weak 2D: tasks=4, grid=4000x4000 (factors 2x2)
Submitted batch job 5833328
  Weak 1D: tasks=4, grid=2000x8000
Submitted batch job 5833329
  Weak 2D: tasks=8, grid=4000x8000 (factors 2x4)
Submitted batch job 5833330
  Weak 1D: tasks=8, grid=2000x16000
Submitted batch job 5833331
  Weak 2D: tasks=16, grid=8000x8000 (factors 4x4)
Submitted batch job 5833332
  Weak 1D: tasks=16, grid=2000x32000
Submitted batch job 5833333
  Weak 2D: tasks=32, grid=8000x16000 (factors 4x8)
Submitted batch job 5833334
  Weak 1D: tasks=32, grid=2000x64000
Submitted batch job 5833335

========================================
  All jobs submitted!
  Check status:  squeue -u $USER
  CSV results:   benchmark_results.csv
========================================
```