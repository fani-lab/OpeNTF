import sys

from cmn.publication import Publication


# output = 'dblp.v12'
output = 'toy'

i2m, m2i, i2s, s2i, i2t, t2i, teams = Publication.read_data(data_path=f'../data/raw/{output}.json', output=f'../data/preprocessed/{output}', topn=None)

i2m, m2i, i2s, s2i, i2t, t2i, teams = Publication.remove_outliers(output=f'../data/preprocessed/{output}', n=2)

# skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub = Publication.get_stats(teams, raw_data_path)
stats = Publication.get_stats(teams, f'../data/preprocessed/{output}')
