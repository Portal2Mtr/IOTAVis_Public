#!/bin/bash

# Run in ./..

# Run 'sudo sysctl -w kernel.perf_event_paranoid=1' before this script in a user-owned directory

algos=("URTS" "MCMC" "EIOTA" "almostURTS" "DT")
#algos=("DT")
trials=($(seq 1 1 10))
waitseq=($(seq 1 1 3)) # Change if need a longer wait time

# Estimate the number of cycles and instructions for the current script
for algo in "${algos[@]}"; do
  for trial in "${trials[@]}"; do
    for wait in "${waitseq[@]}"; do
      echo "Starting next trial in ${wait} seconds..."
      sleep 1
    done
    echo "Running trial #${trial} of ${algo}..."
    #perf stat -e instructions -e cycles -o ./power_analysis/output/"${algo}""${trial}".txt \
    #python3 ./powerscript.py --algo "${algo}" --trial "${trial}"

    python3 ./power_analysis/powerscript.py --algo "${algo}" --trial "${trial}"
    wait
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
    echo "#####################Trial Complete!##############################"
    echo "##################################################################"
    echo "##################################################################"
    echo "##################################################################"
  done
done
echo "Tests complete!"
