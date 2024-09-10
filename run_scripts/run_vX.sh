#!/bin/bash


script_name=$(basename "$0")

set -e  # Exit on any error

exec > "$script_name.log" 2>&1

# COMMAND TO RUN:
# ./run6.sh > run6.log 2>&1 &


# Available datasets
datasets=("toy_dblp" "dblp" "gith" "imdb" "uspt")

# Available dataset paths
dataset_paths=("../data/preprocessed/dblp/toy.dblp.v12.json" "../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3" "../data/preprocessed/gith/gith.data.csv.filtered.mt75.ts3" "../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3" "../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3")
toy_dblp_path=${dataset_paths[0]}
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
run_tag="TEST_wordvec_size"
full_run_name="$script_name"_"$run_tag"

# Select model
model=$nmt_transformer

# Variants array ie. variants=("1" "2" "3")
variants=("wvs-a" "wvs-b" "wvs-c" "wvs-d" "wvs-e")

# Select datasets
datasets=("toy_dblp" "dblp" "gith" "imdb" "uspt")
dataset_path=("$toy_dblp_path" "$dblp_path" "$gith_path" "$imdb_path" "$uspt_path")
is_toy=true

# Select gpus, ie. gpus="0,1,2,3,4,5,6,7"
gpus="6"


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

num_of_variants=${#variants[@]}
num_of_datasets=${#datasets[@]}
total_jobs=$((num_of_variants * num_of_datasets))
current_job=1

# Loop over the datasets
for j in "${!datasets[@]}"; do
  dataset=${datasets[$j]}
  dataset_path=${dataset_paths[$j]}

  # Loop over the variants
  for i in "${!variants[@]}"; do
    # Check if ${variants[$i]} is an empty string
    if [ -z "${variants[$i]}" ]; then
      dash=""
      with_variant_msg=""
    else
      dash="-"
      with_variant_msg=". Variant: ${variants[$i]}"
    fi

    # Get the start time
    start_time=$(date +%s)

    # Run the command with nohup and capture its PID
    nohup python3 -u main.py \
      -data $dataset_path \
      -domain $dataset \
      -model $model \
      -variant "${variants[$i]}" \
      > "../run_logs/${full_run_name}_${dataset}_${model}${dash}${variants[$i]}.log" 2> "../run_logs/${full_run_name}_${dataset}_${model}${dash}${variants[$i]}_errors.log" &
    
    pid=$!

    echo ""
    echo "Job $current_job/$total_jobs"
    echo "\tProcess $pid started. Dataset: ${dataset}. Model: ${model}${with_variant_msg}. GPUs: ${gpus}"

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

    in_minutes=$(($elapsed_time / 60))
    
    echo "\tProcess $pid completed. Duration: $formatted_time ($in_minutes mins)."
    echo ""

    current_job=$((i+1))
  done
done
