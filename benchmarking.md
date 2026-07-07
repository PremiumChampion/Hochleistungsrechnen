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