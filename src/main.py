import sys
import time
from scipy.sparse import load_npz
import pickle

from cmn.team import Team
from cmn.publication import Publication
from dal.data_utils import *
import dnn
import nmt
import sgns

sys.path.extend(['../cmn'])

output = 'dblp.v12'
# output = 'toy'

skill_vecs, member_vecs, i2m, m2i, i2s, s2i, len_teams = Publication.generate_sparse_vectors(raw_data_path=f'../data/raw/{output}.json', output=f'../data/preprocessed/{output}', topn=None)

splits = create_evaluation_splits(len_teams, 5)

dnn.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/{output}/fnn', cmd=['train', 'eval', 'test'])  # ['train', 'test', 'eval']

# nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

# sgns.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/{output}/sgns', cmd=['train', 'eval', 'test'])   # ['test', 'plot', 'eval']
