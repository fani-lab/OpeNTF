#!/bin/bash

# Version info
template_version=1.5

# ------------------------------------------------------------------------------
# WHY THIS SCRIPT?
#
# The reason for this script is to run multiple model(s) and dataset(s) one
# after another without the need to attend in the background. This also avoids
# the need to run multiple terminals and allows for easy monitoring of all
# scripts while logging all terminal outputs in log files (including errors).
#
# That's is the power of this script.
# 
# Note: log files will appear in the "run_logs" directory.
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ONLY EDIT THIS SECTION

# GPU indices to use for training
gpu_indices_to_use="2"

# models to run located in the src/mdl/nmt_models directory
models=("convs2s-final-luna")

# TODO: NOT WORKING YET
# If true, parse datasets from filename (e.g., 1.5-igd-...)
# i = imdb, g = gith, d = dblp, u = uspt
use_dataset_in_filename=false

# Used if use_dataset_in_filename is false
datasets=("uspt")

run_next_script=false
next_script_name="example_next_script.sh"

# INSTRUCTIONS ARE BELOW
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# INSTRUCTIONS
# 
# Recommended script file naming convention:
# <version>-<dataset>-<model>-<tags>.sh
# 
# <version> is the version number of the template
#
# Example:
# 1.5-gith-model1-run1.sh
# 1.5-dblp-model1-run1-set1.sh
#
# Step 1. Duplicate and rename this script using the recommended naming 
#         convention above.
# Step 2. Edit the configurations above.
# Step 3. Run the script by typing "./<script_name>.sh" in the terminal.
#
# ------------------------------------------------------------------------------
# DETAILED INSTRUCTIONS
#
# models=("mode1")
# models to run located in the src/mdl/nmt_models directory
# In that directory, model names are prefixed with 
# "nmt-<dataset_name>-<model_name>-<tags>.yaml"
# i.e., nmt-dblp-model1-afterfix.yaml (then you write "model1" in the models array)
#
# If you want to run a single model models=("model1")
# or you can still set multiple models manually like this:
# models=("model1" "model2" "model3")

# use_dataset_in_filename=false
# If true, parse datasets from filename (e.g., 1.5-igd-...)
# i = imdb, g = gith, d = dblp, u = uspt
# useful approach when running multiple datasets automatically and sequentially
# default: false

# datasets=("dblp")
# i.e., if you want to run a single dataset datasets=("dblp")
# or you can still set multiple datasets manually like this:
# datasets=("dblp" "imdb" "gith" "uspt")
# useful approach when running multiple of these scripts in docker containers in
# parallel (i.e., not sequentially, set one dataset in script and run all in docker
# containers in parallel)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE

# Source common functions
source "$(dirname "${BASH_SOURCE[0]}")/_commons.sh"

# Convert models array to pipe-delimited string
models_string=$(IFS='|'; echo "${models[*]}")

# Convert arrays to pipe-delimited strings
datasets_string=$(IFS='|'; echo "${datasets[*]}")

# Run the main execution logic
nohup bash -c "source \"$(dirname "${BASH_SOURCE[0]}")/_commons.sh\" && \
run_main \
    \"$template_version\" \
    \"$(basename "$0" .sh)\" \
    \"$use_dataset_in_filename\" \
    \"$models_string\" \
    \"$datasets_string\" \
    \"$gpu_indices_to_use\" \
    \"$run_next_script\" \
    \"$next_script_name\"" > "$(basename "$0" .sh).log" 2>&1 &

echo "Script started in background. Check $(basename "$0" .sh).log for progress."