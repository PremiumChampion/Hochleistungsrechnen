#!/usr/bin/env bash

# 1. Define your specific allowed rank counts
ALLOWED_RANKS=(1 2 4 5 8 10 20 25 40 50 100 125 200 250 500 1000)

# 2. Define the node counts you want to test (e.g., power of 2 scaling)
NODE_TESTS=(1 2 4 8)

# 3. Define the physical cores per node (from your previous log: 20)
# We will allow a bit of oversubscription (e.g., up to 2x) for testing purposes
MAX_TASKS_PER_NODE=40 

echo "Starting Scaling Benchmark Scheduler..."
echo "----------------------------------------"

for nodes in "${NODE_TESTS[@]}"; do
    for total_ranks in "${ALLOWED_RANKS[@]}"; do
        
        # RULE 1: Total ranks must be at least equal to number of nodes
        # (To ensure we are actually using all allocated nodes)
        if [ "$total_ranks" -lt "$nodes" ]; then
            continue
        fi

        # RULE 2: Balanced distribution 
        # total_ranks must be evenly divisible by node count
        if (( total_ranks % nodes == 0 )); then
            tasks_per_node=$(( total_ranks / nodes ))

            # RULE 3: Hardware Limit
            # Don't schedule if tasks per node exceed our oversubscription limit
            if [ "$tasks_per_node" -le "$MAX_TASKS_PER_NODE" ]; then
                
                JOB_NAME="M06-N${nodes}-R${total_ranks}"
                
                echo "Submitting: $total_ranks ranks across $nodes nodes ($tasks_per_node tasks/node)"
                
                # We override ntasks-per-node and nodes on the command line
                # This ignores the defaults set inside run.job
                sbatch --nodes="$nodes" \
                       --ntasks-per-node="$tasks_per_node" \
                       --job-name="$JOB_NAME" \
                       run.job
            fi
        fi
    done
done

echo "----------------------------------------"
echo "All valid jobs submitted. Check progress with 'squeue -u $USER'"