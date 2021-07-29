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
output = 'toy'

skill_vecs, member_vecs, i2m, m2i, i2s, s2i, teams = Publication.generate_sparse_vectors(raw_data_path=f'../data/raw/{output}.json', output=f'../data/preprocessed/{output}/', topn=None)

# Arman => the model train-test does not need teams now.
# But just because of n_teams we have to load them all. Try to find a workaround please.
splits = create_evaluation_splits(len(teams), 5)
dnn.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/fnn/{output}/', cmd=['test'])  # ['train', 'test', 'eval']

# nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

# sgns.main(splits, teams, skill_to_index, member_to_index, index_to_skill, index_to_member, output='dblp_v12', cmd=['test', 'eval'])  # ['test', 'plot', 'eval']
