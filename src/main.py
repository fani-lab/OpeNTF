import sys
sys.path.extend(['../cmn'])
from dal.data_utils import *


data_path = "../data/raw/dblp.v12.json"

all_authors, all_docs, input_data, output_data = read_data(data_path)

index_to_author, author_to_index = build_index_authors(all_authors)

index_to_skill, skill_to_index = build_index_skills(all_docs)
