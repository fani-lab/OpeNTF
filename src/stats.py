import sys
import pickle
import matplotlib.pyplot as pyplot
from collections import Counter

from cmn.publication import Publication
output = '../../main/data/preprocessed/dblp/dblp.v12.json'
# output = '../data/preprocessed/dblp/toy.dblp.v12.json'
# with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
#     stats = Publication.get_stats(pickle.load(infile), output, plot=True)

from cmn.movie import Movie
output = '../../main/data/preprocessed/imdb/title.basics.tsv'
# # output = '../data/preprocessed/imdb/toy.title.basics.tsv'
# with open(f'{output}/teamsvecs.pkl', 'rb') as infile:
#     stats = Movie.get_stats(pickle.load(infile), output, plot=True)

from cmn.patent import Patent
from cmn.team import Team
output = '../data/preprocessed/uspt/toy.patent.tsv'
output = '../data/preprocessed/uspt/patent.tsv'
with open(f'{output}/teams.pkl', 'rb') as infile1, open(f'{output}/teamsvecs.pkl', 'rb') as infile2:
    stats = Patent.get_stats(pickle.load(infile1), pickle.load(infile2), output, plot=True)

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
