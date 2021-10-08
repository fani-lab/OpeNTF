import pandas as pd
import preprocess as pp
import stats as st
import getskills as gs

assignee_path = '../data/raw/patent_assignee.tsv'
inventor_path = '../data/raw/patent_inventor.tsv'
location_path = '../data/raw/location.tsv'
save_path = '../data/processed'
dblp_path = '../data/raw/dblp.v12.json'

load_file = ['assignee', 'inventor', 'patent']
#load_file = ['patent']
final_save = './data/processed/patent_final.csv'


pp_obj = pp.Assignee(assignee_path, inventor_path, location_path, save_path, '../data/raw/patent.tsv')
print("Starting Processing for all the tables")
#dummy_save = pp_obj.read_data(load_file)

#pp_obj.extract_abstract()

#pp_obj.build_index()

pp_obj.pack()

# stats_ext = st.Stats(final_save)

# stats_ext.get_stats()

# gskills = gs.Skills()
#
# skills_list = gskills.get_skills(dblp_path)
# print(len(skills_list))


