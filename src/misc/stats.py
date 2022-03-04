import pickle, sys, time
import matplotlib.pyplot as pyplot
from collections import Counter

sys.path.extend(["./"])

import param
from cmn.team import Team

datasets = []
# datasets += ['../data/preprocessed/dblp/dblp.v12.json']
datasets += ['../data/preprocessed/dblp/toy.dblp.v12.json']
# datasets += ['../data/preprocessed/imdb/title.basics.tsv']
datasets += ['../data/preprocessed/imdb/toy.title.basics.tsv']
# datasets += ['../data/preprocessed/uspt/patent.tsv']
datasets += ['../data/preprocessed/uspt/toy.patent.tsv']

for dataset in datasets:
    start = time.time()
    with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile:
        stats = Team.get_stats(pickle.load(infile), dataset, cache=False, plot=True, plot_title='uspt')
    end = time.time()
    print(end - start)

# from cmn.patent import Patent
# output = '../data/preprocessed/uspt/patent.tsv'
# # output = '../data/preprocessed/uspt/toy.patent.tsv'
# with open(f'{output}/teams.pkl', 'rb') as infile1, open(f'{output}/teamsvecs.pkl', 'rb') as infile2:
#     stats = Patent.get_stats(pickle.load(infile1), pickle.load(infile2), output, plot=True)


#######################
# stat generation and figure for parallel sparse matrix creation
#######################
# import re
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
# #     # for i in range(1000, 1000000, 1000):
# #     #     param.settings['data']['domain']['dblp']['nrow'] = i
# #     #     Publication.generate_sparse_vectors('./../data/raw/dblp/dblp.v12.json', f"./serial/{param.settings['data']['domain']['dblp']['nrow']}", 0, param.settings['data'])
# #
#     with open('../data/parallel.log', "r", encoding='utf-8') as p, open('../data/serial.log', "r", encoding='utf-8') as s:
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
#     plt.rcParams.update({'font.family': 'Consolas'})
#     plt.figure()
#     fig = plt.figure(figsize=(3, 3))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_xlabel('#teams (papers in dblp.v12)')
#     ax.set_ylabel('time (second)')
#     # plt.title(f'intel xeon e-2236 3.40 ghz (12 cores), 64 gb memory, bucket size = 500')
#     ax.xaxis.set_tick_params(size=2, direction='in')
#     ax.yaxis.set_tick_params(size=2, direction='in')
#     ax.xaxis.get_label().set_size(12)
#     ax.yaxis.get_label().set_size(12)
#     ax.grid(True, color="#93a1a1", alpha=0.3)
#     ax.set_facecolor('whitesmoke')
#
#     plt.plot(pkl_lines_par.keys(), pkl_lines_par.values(), linestyle='-', label='loading raw data')
#     plt.plot(vec_lines_par.keys(), vec_lines_par.values(), linestyle='--', label='parallel prep.')
#     # plt.plot(pkl_lines_ser.keys(), pkl_lines_ser.values(), label='pickling raw data')
#     plt.plot(vec_lines_ser.keys(), vec_lines_ser.values(), linestyle='-.', label='sequential prep.')
#
#     # plt.loglog(pkl_lines_par.keys(), pkl_lines_par.values(), linestyle='-', label='loading raw data')
#     # plt.loglog(vec_lines_par.keys(), vec_lines_par.values(), linestyle='--', label='parallel prep.')
#     # # plt.loglog(pkl_lines_ser.keys(), pkl_lines_ser.values(), label='pickling raw data')
#     # plt.loglog(vec_lines_ser.keys(), vec_lines_ser.values(), linestyle='-.', label='sequential prep.')
#     plt.legend()
#     # plt.setp(ax.get_xticklabels(), fontsize=14)
#     # plt.setp(ax.get_yticklabels(), fontsize=14)
#     plt.show()
#     fig.savefig(f'../data/speedup.pdf', dpi=100, bbox_inches='tight')
#
#     import numpy as np
#     print(np.polyfit(list(pkl_lines_par.keys()), list(pkl_lines_par.values()), 1))
#     print(np.polyfit(list(vec_lines_par.keys()), list(vec_lines_par.values()), 1))
#     print(np.polyfit(list(vec_lines_ser.keys()), list(vec_lines_ser.values()), 1))
#     #
#     # Yann LeCun : 2053214915
#     # Geoffrey E.Hinton : 563069026
#     # Yoshua Bengio : 161269817
#     pass

