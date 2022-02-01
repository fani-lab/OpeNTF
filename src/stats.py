import param

import sys
import pickle
import matplotlib.pyplot as pyplot
from collections import Counter

from cmn.publication import Publication
# output = '../../main/data/preprocessed/dblp/dblp.v12.json'
# output = '../data/preprocessed/dblp/toy.dblp.v12.json'
# with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
#     stats = Publication.get_stats(pickle.load(infile), output, plot=True)

# from cmn.movie import Movie
# output = '../../main/data/preprocessed/imdb/title.basics.tsv'
# output = '../data/preprocessed/imdb/toy.title.basics.tsv'
# with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
#     stats = Movie.get_stats(pickle.load(infile), output, plot=True)

# from cmn.patent import Patent
# from cmn.team import Team
# output = '../data/preprocessed/uspt/toy.patent.tsv'
# # output = '../data/preprocessed/uspt/patent.tsv'
# with open(f'{output}/teams.pkl', 'rb') as infile1, open(f'{output}/teamsvecs.pkl', 'rb') as infile2:
#     stats = Patent.get_stats(pickle.load(infile1), pickle.load(infile2), output, plot=True)

# with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
#     teamsvecs = pickle.load(infile)
#     teamids, skillvecs, membervecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member']
#
#     #checking whether there are empty skill sets. The issue raised by https://github.com/fani-lab/neural_team_formation/issues/79
#     row_sums = skillvecs.sum(axis=1)
#     nteams_nskills = Counter(row_sums.A1.astype(int))
#     stats = {}
#     stats['nteams_nskills'] = {k: v for k, v in sorted(nteams_nskills.items(), key=lambda item: item[1], reverse=True)}
#     print(stats['nteams_nskills'][0])
#     #in dblp, there are 85 teams with no skills!
#
#
#     # skill_member = (skillvecs.transpose() @ membervecs)
#     # pyplot.figure(figsize=(50, 50), dpi=300)
#     # color_map = pyplot.imshow(skill_member[:1000, :2000].astype(int).todense())
#     # color_map.set_cmap("Blues_r")
#     # pyplot.colorbar()
#     # pyplot.show()
#
#     # pyplot.figure(dpi=300)
#     # pyplot.spy(skillvecs[4000000: 4000100, :100], markersize=1)
#     # pyplot.show()
#
#     # n = 6
#     # fig, axs = pyplot.subplots(1, n)
#     # for i in range(n):
#     #     axs[i].set_xticklabels([])
#     #     axs[i].set_yticklabels([])
#     #     axs[i].spy(skillvecs[i * 1000000 : (i+1) * 1000000, :], markersize=1)
#     # pyplot.show()
#     #
#     # # pyplot.spy(membervecs[:80000,:80000], markersize=1)
#     # # pyplot.show()
#     #
#     # #the result is dense!
#     # n = 10
#     # for j in range(1,6):
#     #     fig, axs = pyplot.subplots(n)
#     #     for i in range(n):
#     #         axs[i].set_xticklabels([])
#     #         axs[i].set_yticklabels([])
#     #         axs[i].spy(skill_member[:, (j - 1) * 1000000 : j * 1000000], precision=2**i, markersize=1)
#     #     pyplot.show()
#

#######################
# stat generation and figure for parallel sparse matrix creation
#######################
# import re
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     # for i in range(1000, 1000000, 1000):
#     #     param.settings['data']['domain']['dblp']['nrow'] = i
#     #     Publication.generate_sparse_vectors('./../data/raw/dblp/dblp.v12.json', f"./serial/{param.settings['data']['domain']['dblp']['nrow']}", 0, param.settings['data'])
#
#     plt.figure()
#     with open('./parallel.log', "r", encoding='utf-8') as p, open('./serial_.log', "r", encoding='utf-8') as s:
#         pkl_lines_par = {}; vec_lines_par = {}
#         pkl_lines_ser = {}; vec_lines_ser = {}
#         for i, line in enumerate(p):
#             if i % 9 in {7}:
#                 numbers = [float(s) for s in re.findall(r'\d+\.?\d*', line)]
#                 pkl_lines_par[numbers[1]] = numbers[0]
#             if i % 9 in {8}:
#                 vec_lines_par[numbers[1]] = float(re.findall(r'\d+\.?\d*', line)[0])
#         for i, line in enumerate(s):
#             if i % 9 in {7}:
#                 numbers = [float(s) for s in re.findall(r'\d+\.?\d*', line)]
#                 pkl_lines_ser[numbers[1]] = numbers[0]
#             if i % 9 in {8}:
#                 vec_lines_ser[numbers[1]] = float(re.findall(r'\d+\.?\d*', line)[0])
#
#     plt.xlabel('#papers (teams)')
#     plt.ylabel('time (second)')
#     plt.title(f'intel xeon e-2236 3.40 ghz (12 cores), 64 gb memory')
#     plt.plot(pkl_lines_par.keys(), pkl_lines_par.values(), label='pickling raw data (dblp.v12)')
#     plt.plot(vec_lines_par.keys(), vec_lines_par.values(), label='creating sparse matrix in parallel')
#     # plt.plot(pkl_lines_ser.keys(), pkl_lines_ser.values(), label='pickling raw data')
#     plt.plot(vec_lines_ser.keys(), vec_lines_ser.values(), label='creating sparse matrix in sequential')
#     plt.legend()
#     plt.show()

