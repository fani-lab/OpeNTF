import sys

from cmn.publication import Publication


# output = 'dblp.v12.json'
output = 'toy.json'

import pickle
with open(f'../data/preprocessed/{output}/teamsvecs_filtered.pkl', 'rb') as infile:
    stats = Publication.get_stats(pickle.load(infile), f'../data/preprocessed/{output}', plot=True)

with open(f'../data/preprocessed/{output}/teamsvecs.pkl', 'rb') as infile:
    stats = Publication.get_stats(pickle.load(infile), f'../data/preprocessed/{output}', plot=True)

