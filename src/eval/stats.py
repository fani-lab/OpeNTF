import sys
import pickle
sys.path.extend(['../cmn'])

from publication import Publication
# output = '../data/preprocessed/dblp/dblp.v12.json'
# output = '../data/preprocessed/dblp/dblp.v12.filtered'
output = '../data/preprocessed/dblp/toy.json'
# output = '../data/preprocessed/dblp/toy.json.filtered'
with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
    stats = Publication.get_stats(pickle.load(infile), output, plot=True)

from movie import Movie
# output = '../data/preprocessed/imdb/title.basics.tsv'
# output = '../data/preprocessed/imdb/title.basics.tsv.filtered'
output = '../data/preprocessed/imdb/toy.title.basics.tsv'
# output = '../data/preprocessed/imdb/toy.title.basics.tsv.filtered'
with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
    stats = Movie.get_stats(pickle.load(infile), output, plot=True)

