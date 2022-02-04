#!/bin/bash

echo "Starting workers..."
workers=($(seq 1 1 20))

# Estimate the number of cycles and instructions for the current script
for worker in "${workers[@]}"; do
  echo "Started worker ${worker}!"
  tmux new-session -d -s "optuna${worker}" "python3 intelligent_tip/dttraining.py --seed=${worker} --runStudy"
  sleep 3
done

echo "Workers running!"

# Run script below to reset study
#for worker in "${worker[@]}"; do
#  tmux kill-session -t 'optuna${worker}'
#done
#source activate clusterProj
#python3 intelligent_tip/resetserver.py
