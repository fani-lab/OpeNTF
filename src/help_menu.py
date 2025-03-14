#!/usr/bin/env python3
"""
Simple script to print the help menu in the desired format for OpeNTF
"""

HELP_TEXT = """OpeNTF: Open Neural Team Formation

Required:
   -i INPUT, --input INPUT
	Input file or folder

   -d DOMAIN, --domain DOMAIN
	Domain of the dataset. Options: dblp, gith, imdb, uspt


Optionals:
   -m MODEL, --model MODEL
	Model to perform the task, or the type of the experiments to run, e.g., random, heuristic, expert, etc. If not provided, process will stop after data loading.

   -train TRAIN, --train TRAIN
	Whether to train the model

   -filter FILTER, --filter FILTER
	Whether to filter data: zero: no filtering, one: filter zero degree nodes, two: filter one degree nodes

   -future FUTURE, --future FUTURE
	Forecast future teams: zero: no need to forecast future teams, one: predict future teams

   -fair FAIR, --fair FAIR
	Apply fairness to model

   -o OUTPUT, --output OUTPUT
	Output file or folder

   -gpus GPUS, --gpus GPUS
	CUDA Visible GPUs

   -t THREADS, --threads THREADS
	Number of threads to use for parallel processing (0 for auto, defaults to 75% of available CPU cores)

   -b BATCH_SIZE, --batch-size BATCH_SIZE
	Batch size for processing large datasets (default: IMDB: 10000, DBLP: 10000, GITH: 1000, USPT: 5000)
"""

if __name__ == "__main__":
    print(HELP_TEXT) 