#!/bin/bash

##setup the env
#module load python/3.8;
#module load scipy-stack;
##virtualenv --no-download env_opentf;
#source env_opentf/bin/activate;
#pip install --no-index --upgrade pip;

#SBATCH --time=20:00:00
#SBATCH --account=def-hfani
#SBATCH --gpus-per-node=2
#SBATCH --mem=64000M
#SBATCH --mail-user=hfani@uwindsor.ca
#SBATCH --mail-type=ALL
#nvidia-smi

python main.py -data ../data/raw/dblp/dblp.v12.json -domain dblp -model nmt -filter 1

