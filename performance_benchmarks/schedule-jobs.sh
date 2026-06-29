#!/usr/bin/env bash

MAX=1000

for no_nodes in $(seq 1 "$MAX"); do
  for no_tasks_node in $(seq "$no_nodes" "$no_nodes" "$MAX"); do
    if (( MAX % no_tasks_node == 0 )); then
      sbatch --nodes="$no_nodes" --ntasks-per-node="$no_tasks_node" \
        -J "milestone06-${no_nodes}-${no_tasks_node}" run.job
    fi
  done
done