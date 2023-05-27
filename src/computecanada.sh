#!/bin/bash

##setup the env
#module load python/3.8;
#module load scipy-stack;
##virtualenv --no-download env_opentf;
#source env_opentf/bin/activate;
#pip install --no-index --upgrade pip;

#SBATCH --time=2-00
#SBATCH --account=def-hfani
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --mail-user=hfani@uwindsor.ca
#SBATCH --mail-type=ALL
#nvidia-smi

python -u main.py -data ../data/raw/dblp/dblp.v12.json -domain dblp -model bnn -filter 1
#python -u main.py -data ../data/raw/imdb/title.basics.tsv -domain imdb -model bnn -filter 1
#python -u main.py -data ../data/raw/uspt/patent.tsv -domain uspt -model bnn -filter 1