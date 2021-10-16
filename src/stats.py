import sys
import pickle

from cmn.publication import Publication

# output = '../data/preprocessed/dblp/dblp.v12.json'
# output = '../data/preprocessed/dblp/toy.json'
# with open(f'{output}.filtered/teamsvecs.pkl', 'rb') as infile:
#     stats = Publication.get_stats(pickle.load(infile), f'../data/preprocessed/{output}', plot=True)
#
# with open(f'../data/preprocessed/dblp/{output}/teamsvecs.pkl', 'rb') as infile:
#     stats = Publication.get_stats(pickle.load(infile), f'../data/preprocessed/{output}', plot=True)

from cmn.movie import Movie

output = '../data/preprocessed/imdb/toy.title.basics.tsv'
# with open(f'{output}.filtered/teamsvecs.pkl', 'rb') as infile:
#     stats = Movie.get_stats(pickle.load(infile), f'{output}', plot=True)

with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
    stats = Movie.get_stats(pickle.load(infile), f'{output}', plot=True)

