#!/bin/bash

set -e  # Exit on any error




# Available datasets
datasets=("dblp" "gith" "imdb" "uspt")
dblp=${datasets[0]}
gith=${datasets[1]}
imdb=${datasets[2]}
uspt=${datasets[3]}

# Available dataset paths
dataset_paths=("../data/preprocessed/dblp/toy.dblp.v12.json" "../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3" "../data/preprocessed/gith/gith.data.csv.filtered.mt10.ts3" "../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3" "../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3")
dblp_toy_path=${dataset_paths[0]}
dblp_path=${dataset_paths[1]}
gith_path=${dataset_paths[2]}
imdb_path=${dataset_paths[3]}
uspt_path=${dataset_paths[4]}

# Available models
models=("nmt_convs2s" "nmt_rnn" "nmt_transformer")
nmt_convs2s=${models[0]}
nmt_rnn=${models[1]}
nmt_transformer=${models[2]}

# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------------------------

# Run number
run_num=0

# Array of hyperparameters
# Add hyperparameters here in the array ie. hyp_num=("1" "2" "3")
hyp_num=("")

# Select dataset
dataset=$imdb
dataset_path=$imdb_path
is_toy=false

# Select model
model=$nmt_convs2s

# ------------------------------------------------------------------------------
# END CONFIGURATIONS
# ------------------------------------------------------------------------------


# Log file for all run times
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
times_dir="${SCRIPT_DIR}/../run_times"
logs_dir="${SCRIPT_DIR}/../run_logs"

# Ensure the directories exist
mkdir -p ../run_logs

cd ../src

# Loop over the output names
for i in "${!hyp_num[@]}"; do
  echo ""
  echo "Processing with hyperparameter $((i+1)): '${hyp_num[$i]}'"

  # Get the start time
  start_time=$(date +%s)

  # Check if ${hyp_num[$i]} is an empty string
  if [ -z "${hyp_num[$i]}" ]; then
    dash=""
    with_hyp_msg=""
  else
    dash="_"
    with_hyp_msg=" with hyperparam ${hyp_num[$i]}"
  fi

  if [ $is_toy = true ]; then
    is_toy_msg="_toy"
  else
    is_toy_msg=""
  fi
  
  # Run the command with nohup and capture its PID
  echo "Running: nohup python3 -u main.py ..."
  nohup python3 -u main.py \
    -data $dataset_path \
    -domain $dataset \
    -model $model${dash}${hyp_num[$i]} \
    > "../run_logs/run${run_num}${is_toy_msg}_${dataset}_${model}${dash}${hyp_num[$i]}.log" 2> "../run_logs/run${run_num}${is_toy_msg}_${dataset}_${model}${dash}${hyp_num[$i]}_errors.log" &
  
  pid=$!

  echo ""
  echo "Started process $pid for run ${run_num}${with_hyp_msg}"
  echo ""

  # Wait for the process to complete
  wait $pid

  # Get the end time
  end_time=$(date +%s)
  
  # Calculate the elapsed time in seconds
  elapsed_time=$(($end_time - $start_time))
  
  # Convert elapsed time into hours, minutes, and seconds
  hours=$(($elapsed_time / 3600))
  minutes=$(($elapsed_time % 3600 / 60))
  seconds=$(($elapsed_time % 60))
  
  # Format the elapsed time as Xh Xm Xs
  formatted_time="${hours}h ${minutes}m ${seconds}s"
  
  echo "Process for run ${run_num}${with_hyp_msg} completed. Duration: $formatted_time."
  echo ""
done