#!/bin/bash

template_version=1.4.1
script_name=$(basename "$0" .sh)

# Ensure the script runs in the background with nohup and redirects output to a log file
if [[ "$1" != "--nohup" ]]; then
  nohup "$0" --nohup "$@" > "${script_name}.log" 2>&1 &
  exit
fi

set -e  # Exit on any error

# COMMAND TO RUN:
# ./scriptname.sh

# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------------------------

models=("transformer-final")
datasets=("uspt")
gpus="6,7"

# ------------------------------------------------------------------------------
# END CONFIGURATIONS
# ------------------------------------------------------------------------------

declare -A dataset_paths
dataset_paths["toy_dblp"]="../data/preprocessed/dblp/toy.dblp.v12.json"
dataset_paths["dblp"]="../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3"
dataset_paths["gith"]="../data/preprocessed/gith/gith.data.csv.filtered.mt75.ts3"
dataset_paths["imdb"]="../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3"
dataset_paths["uspt"]="../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3"

# Log file for all run times
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
times_dir="${SCRIPT_DIR}/../run_times"
logs_dir="${SCRIPT_DIR}/../run_logs"

# get length of models array
num_models=${#models[@]}

# get length of datasets array
num_datasets=${#datasets[@]}

# total jobs to run
jobs=$((num_models * num_datasets))
job_num=1

# Ensure the directories exist
mkdir -p ../run_logs
cd ../src
echo -e "Template\t: v${template_version}"
echo -e "Script\t\t: ${script_name}.sh"
echo -e "Jobs\t\t: ${jobs}"

# loop over models and each model over datasets
for model in "${models[@]}"; do
  for dataset in "${!datasets[@]}"; do

    current_dataset="${datasets[$dataset]}"
    current_dataset_path="${dataset_paths[$current_dataset]}"
    current_model="${current_dataset}_${model}"

    # if current_dataset has "toy_" prefix, then take out the prefix and assign the dataset name to domain variable
    if [[ $current_dataset == *"toy_"* ]]; then
      domain=${current_dataset#"toy_"}
    else
      domain=$current_dataset
    fi

    echo ""
    # Get the start time
    start_time=$(date +%s)
    start_time_est=$(date -u -d "-3 hours -33 minutes" +"%Y-%m-%d %H:%M:%S")
    
    # Run the command with nohup and capture its PID
    nohup python3 -u main.py \
      -data $current_dataset_path \
      -domain $domain \
      -model "nmt_${current_model}" \
      -gpus $gpus \
      > "../run_logs/${current_model}.log" 2> "../run_logs/${current_model}_errors.log" &
    
    pid=$!
    
    echo "[Job $job_num/$jobs] Process $pid. Model: ${current_model}. Dataset: ${current_dataset}."
    echo -e "\tStarted at\t: ${start_time_est} EST"
    
    # Wait for the process to complete
    wait $pid
    
    # Get the end time
    end_time=$(date +%s)
    end_time_est=$(date -u -d "-3 hours -33 minutes" +"%Y-%m-%d %H:%M:%S")
     echo -e "\tEnded at\t: ${end_time_est} EST"
    
    # Calculate the elapsed time in seconds
    elapsed_time=$(($end_time - $start_time))

    # Convert elapsed time into hours, minutes, and seconds
    hours=$(($elapsed_time / 3600))
    minutes=$(($elapsed_time % 3600 / 60))
    seconds=$(($elapsed_time % 60))
    
    # Format the elapsed time as Xh Xm Xs
    formatted_time="${hours}h ${minutes}m ${seconds}s"

    in_minutes=$(($elapsed_time / 60))
    job_num=$((job_num + 1))

    echo -e "\tElapsed\t\t: ${formatted_time} ($in_minutes mins)."
  done
done

echo ""
echo "All ${jobs} jobs completed." 