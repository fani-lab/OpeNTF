#!/bin/bash

# Dataset paths configuration
declare -A dataset_paths
dataset_paths["toy_dblp"]="../data/preprocessed/dblp/toy.dblp.v12.json"
dataset_paths["dblp"]="../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3"
dataset_paths["gith"]="../data/preprocessed/gith/gith.data.csv.filtered.mt75.ts3"
dataset_paths["imdb"]="../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3"
dataset_paths["uspt"]="../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3" 