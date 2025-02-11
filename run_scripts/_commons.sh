#!/bin/bash

# Suppress FutureWarnings
export PYTHONWARNINGS="ignore::FutureWarning"

# Dataset paths configuration
declare -A dataset_paths
dataset_paths["toy_dblp"]="../data/preprocessed/dblp/toy.dblp.v12.json"
dataset_paths["dblp"]="../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3"
dataset_paths["gith"]="../data/preprocessed/gith/gith.data.csv.filtered.mt75.ts3"
dataset_paths["imdb"]="../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3"
dataset_paths["uspt"]="../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3"

# Parse auto datasets from script name
parse_auto_datasets() {
    local script_name=$1
    local use_from_filename=$2
    local _datasets=$(echo "$script_name" | cut -d'-' -f2)
    
    # Check if _datasets contains only i,g,d,u characters
    if [[ $use_from_filename == true ]] && [[ $_datasets =~ ^[igdu]+$ ]]; then
        use_auto_datasets=true
        
        # Initialize empty auto_datasets array
        auto_datasets=()
        
        # Loop through each character in _datasets
        for (( i=0; i<${#_datasets}; i++ )); do
            char="${_datasets:$i:1}"
            case $char in
                i) auto_datasets+=("imdb") ;;
                g) auto_datasets+=("gith") ;;
                u) auto_datasets+=("uspt") ;;
                d) auto_datasets+=("dblp") ;;
            esac
        done
        
        datasets=("${auto_datasets[@]}")
    else
        use_auto_datasets=false
        datasets=("${manual_datasets[@]}")
    fi
}

# Common directory setup
setup_directories() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    logs_dir="${SCRIPT_DIR}/../run_logs"
    mkdir -p ../run_logs
    cd ../src
}

# Print script info
print_script_info() {
    local template_version=$1
    local script_name=$2
    local jobs=$3
    
    echo -e "Template\t: v${template_version}"
    echo -e "Script\t\t: ${script_name}.sh"
    echo -e "Jobs\t\t: ${jobs}"
}

# Format and print job status
print_job_status() {
    local job_num=$1
    local total_jobs=$2
    local pid=$3
    local model=$4
    local dataset=$5
    
    echo ""
    echo "[Job $job_num/$total_jobs] Process $pid. Model: ${model}. Dataset: ${dataset}."
}

# Get and print timestamp in EST
get_time_est() {
    TZ="America/New_York" date +"%Y-%m-%d %H:%M:%S"
}

# Print timestamps
print_timestamps() {
    local action=$1  # "Started" or "Ended"
    local timestamp=$(get_time_est)
    echo -e "\t${action} at\t: ${timestamp} EST"
}

# Calculate and print elapsed time
print_elapsed_time() {
    local start_time=$1
    local end_time=$2
    
    local elapsed_time=$((end_time - start_time))
    local hours=$((elapsed_time / 3600))
    local minutes=$((elapsed_time % 3600 / 60))
    local seconds=$((elapsed_time % 60))
    local in_minutes=$((elapsed_time / 60))
    
    local formatted_time="${hours}h ${minutes}m ${seconds}s"
    echo -e "\tElapsed\t\t: ${formatted_time} (${in_minutes} mins)."
}

# Run command with logging
run_command() {
    local current_dataset_path=$1
    local domain=$2
    local current_model=$3
    local gpus=$4
    
    nohup python3 -u main.py \
        -data $current_dataset_path \
        -domain $domain \
        -model "nmt_${current_model}" \
        -gpus $gpus \
        > "../run_logs/${current_model}.log" 2> "../run_logs/${current_model}_errors.log"
}

# Main execution logic
run_main() {
    local template_version=$1
    local script_name=$2
    local use_dataset_in_filename=$3
    local models_string=$4  # Receiving pipe-delimited string
    local datasets_string=$5  # Receiving pipe-delimited string
    local gpu_indices=$6
    local run_next_script=$7
    local next_script_name=$8

    # Split strings into arrays
    IFS='|' read -ra models <<< "$models_string"
    IFS='|' read -ra manual_datasets <<< "$datasets_string"

    # Remove the nohup check since we'll handle background execution differently
    set -e  # Exit on any error

    # Parse datasets from script name
    parse_auto_datasets "$script_name" "$use_dataset_in_filename"

    # Setup directories
    setup_directories

    # get length of models array
    num_models=${#models[@]}
    num_datasets=${#datasets[@]}
    jobs=$((num_models * num_datasets))
    job_num=1

    # Print script info
    print_script_info "$template_version" "$script_name" "$jobs"

    start_time=$(date +%s)

    # loop over models and each model over datasets
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            current_dataset="${dataset}"
            current_dataset_path="${dataset_paths[$current_dataset]}"
            current_model="${current_dataset}_${model}"

            # Create separate log files for each model
            log_file="../run_logs/${current_model}.log"
            error_log="../run_logs/${current_model}_errors.log"

            # if current_dataset has "toy_" prefix, then take out the prefix
            if [[ $current_dataset == *"toy_"* ]]; then
                domain=${current_dataset#"toy_"}
            else
                domain=$current_dataset
            fi

            # Print job status
            print_job_status "$job_num" "$jobs" "$$" "$current_model" "$current_dataset"
            
            # Print start time
            print_timestamps "Started"
            
            # Run command with model-specific log files
            python3 -u main.py \
                -data "$current_dataset_path" \
                -domain "$domain" \
                -model "nmt_${current_model}" \
                -gpus "$gpu_indices" \
                > "$log_file" 2> "$error_log"
            
            # Get the end time
            end_time=$(date +%s)
            print_timestamps "Ended"
            
            # Print elapsed time
            print_elapsed_time "$start_time" "$end_time"

            job_num=$((job_num + 1))
        done
    done

    echo ""
    echo "All ${jobs} jobs completed."

    # Run the next .sh file if the flag is set
    if [ "$run_next_script" = true ]; then
        echo ""
        echo "Running the next .sh file..."
        ./"$next_script_name"
    fi
} 