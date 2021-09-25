import sys

from cmn.publication import Publication


# output = 'dblp.v12.json'
output = 'toy.json'

# i2m, m2i, i2s, s2i, i2t, t2i, teams = Publication.read_data(data_path=f'../data/raw/{output}.json', output=f'../data/preprocessed/{output}', topn=None)
#
# i2m, m2i, i2s, s2i, i2t, t2i, teams = Publication.remove_outliers(output=f'../data/preprocessed/{output}', n=2)

import pickle
with open(f'../data/preprocessed/{output}/teamsvecs.pkl', 'rb') as infile:
    stats = Publication.get_stats(pickle.load(infile), f'../data/preprocessed/{output}', plot=True)
