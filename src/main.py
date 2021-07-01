import sys
from cmn.team import Team
from cmn.publication import Publication
from dal.data_utils import *
import dnn
import nmt
import sgns

sys.path.extend(['../cmn'])
# raw_data_path = "../data/raw/dblp.v12.json"
raw_data_path = "../data/raw/named_toy.json"

all_members, teams = Publication.read_data(raw_data_path, topn=None)
index_to_member, member_to_index = Team.build_index_members(all_members)
index_to_skill, skill_to_index = Team.build_index_skills(teams)
splits = create_evaluation_splits(len(teams), 2)

dnn.main(splits, teams, skill_to_index, member_to_index, index_to_skill, index_to_member, output='named_toy', cmd=['train', 'test', 'plot', 'eval'])#['train', 'test', 'plot', 'eval']

# nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

# sgns.main(splits, teams, skill_to_index, member_to_index, index_to_skill, index_to_member, cmd=['eval', 'test'])
