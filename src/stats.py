import sys

from cmn.publication import Publication


output = 'dblp.v12'
output = 'toy'

teams = Publication.generate_sparse_vectors(raw_data_path=f'../data/raw/{output}.json', output=f'../data/preprocessed/{output}/', topn=None)[-1]

# skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub = Publication.get_stats(teams, raw_data_path)
stats = Publication.get_stats(teams, f'../data/preprocessed/{output}/stats.json')