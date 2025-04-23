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

python -u main.py data.domain=cmn.publication.Publication data.source=../data/dblp/toy.dblp.v12.json data.output=../output/dblp/toy.dblp.v12.json ~data.filter

#python -u main.py data.domain=cmn.movie.Movie data.source=../data/imdb/toy.title.basics.tsv data.output=../output/imdb/toy.title.basics.tsv ~data.filter
#python -u main.py data.domain=cmn.repository.Repository data.source=../data/gith/toy.repos.csv data.output=../output/gith/toy.repos.csv ~data.filter
#python -u main.py data.domain=cmn.patent.Patent data.source=../data/uspt/toy.patent.tsv data.output=../output/uspt/toy.patent.tsv ~data.filter
