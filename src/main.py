import sys
sys.path.extend(['../cmn'])
from cmn.team import Team
from cmn.document import Document
from dal.data_utils import *
import dnn
import nmt
import sgns

raw_data_path = "../data/raw/named_toy_dataset.json"

all_members, teams, input_data, output_data = Document.read_data(raw_data_path, topn=None)
index_to_member, member_to_index = Team.build_index_members(all_members)
index_to_skill, skill_to_index = Team.build_index_skills(teams)
splits = create_evaluation_splits(len(teams), 5)

dnn.main(splits, teams, skill_to_index, member_to_index, index_to_skill, index_to_member, cmd=['train', 'plot', 'eval', 'test'])

# nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

# sgns.main(splits, teams, skill_to_index, member_to_index, index_to_skill, index_to_member, cmd=['train', 'plot', 'eval', 'test'])
