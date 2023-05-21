import pickle, sys, time
from turtle import color
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import json
import numpy as np
sys.path.extend(["./"])

import param
from cmn.team import Team

datasets = []
# datasets += ['../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/dblp/toy.dblp.v12.json']
# datasets += ['../data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/imdb/toy.title.basics.tsv']
# datasets += ['../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/uspt/toy.patent.tsv']
datasets += ['../data/preprocessed/gith/data.csv']
# datasets += ['../data/preprocessed/gith/data.csv.filtered.mt10.ts3']
# datasets += ['../data/preprocessed/gith/toy.data.csv']

for dataset in datasets:
    start = time.time()
    with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile:
        obj = pd.read_pickle(dataset+'/indexes.pkl')
        stats = Team.get_stats(pickle.load(infile), obj, dataset, cache=False, plot=True, plot_title='uspt')
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

#     plt.plot(pkl_lines_par.keys(), pkl_lines_par.values(), linestyle='-', label='loading raw data')
#     plt.plot(vec_lines_par.keys(), vec_lines_par.values(), linestyle='--', label='parallel prep.')
#     # plt.plot(pkl_lines_ser.keys(), pkl_lines_ser.values(), label='pickling raw data')
#     plt.plot(vec_lines_ser.keys(), vec_lines_ser.values(), linestyle='-.', label='sequential prep.')

#     # plt.loglog(pkl_lines_par.keys(), pkl_lines_par.values(), linestyle='-', label='loading raw data')
#     # plt.loglog(vec_lines_par.keys(), vec_lines_par.values(), linestyle='--', label='parallel prep.')
#     # # plt.loglog(pkl_lines_ser.keys(), pkl_lines_ser.values(), label='pickling raw data')
#     # plt.loglog(vec_lines_ser.keys(), vec_lines_ser.values(), linestyle='-.', label='sequential prep.')
#     plt.legend()
#     # plt.setp(ax.get_xticklabels(), fontsize=14)
#     # plt.setp(ax.get_yticklabels(), fontsize=14)
#     plt.show()
#     fig.savefig(f'../data/speedup.pdf', dpi=100, bbox_inches='tight')

#     import numpy as np
#     print(np.polyfit(list(pkl_lines_par.keys()), list(pkl_lines_par.values()), 1))
#     print(np.polyfit(list(vec_lines_par.keys()), list(vec_lines_par.values()), 1))
#     print(np.polyfit(list(vec_lines_ser.keys()), list(vec_lines_ser.values()), 1))
#     #
#     # Yann LeCun : 2053214915
#     # Geoffrey E.Hinton : 563069026
#     # Yoshua Bengio : 161269817
#     pass

######################
# Temporal Stats
######################
def get_dist_teams_over_years_with_ind(dataset):
    dname = dataset.split('/')[-2]
    with open(f'{dataset}/indexes.pkl', 'rb') as infile:
        indexes = pickle.load(infile)
    year_idx = indexes['i2y']
    nteams = len(indexes['i2t'])
    print(year_idx)
    year_nteams = {}
    for i in range(1, len(year_idx)):
        year_nteams[year_idx[i-1][1]] = year_idx[i][0] - year_idx[i-1][0]
    year_nteams[year_idx[-1][1]] = nteams - year_idx[-1][0]
    print(year_nteams)
    yc = dict(sorted(year_nteams.items()))
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(*zip(*yc.items()), marker='x', linestyle='None', markeredgecolor='b')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ax.set_xticks(list(range(year_idx[0][1], year_idx[-1][1], 25)))
    min_of_nteams = min(yc.values())
    max_of_nteams = max(yc.values())
    ax.set_yticks(list(range(0, max_of_nteams, 100)))
    ax.grid(True, color="#93a1a1", alpha=0.3)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.minorticks_off()
    ax.xaxis.set_tick_params(size=2, direction='in')
    ax.yaxis.set_tick_params(size=2, direction='in')
    ax.set_xlabel('years')
    ax.set_ylabel('#teams')
    ax.set_facecolor('whitesmoke')
    fig.savefig(f"{dataset}/{dname}_time_distribution.pdf", dpi=100, bbox_inches='tight')
    plt.show()

# for dataset in datasets:
    # get_dist_teams_over_years_with_ind(dataset)

def hmap_after_year(dataset_path, values, year):
    dname = dataset_path.split('/')[-2]
    with open(f'{dataset_path}/teamsvecs.pkl', 'rb') as infile:
        teamsvecs = pickle.load(infile)
    with open(f'{dataset_path}/indexes.pkl', 'rb') as infile:
        indexes = pickle.load(infile)
    year_idx = indexes['i2y']
    yid = []
    for i in range(len(year_idx)):
        if year_idx[i][1] < year:
            continue
        yid.append(year_idx[i])
    value_year = np.empty((len(yid), teamsvecs[values].shape[1]))
    for i in range(1, len(yid)):
        if yid[i][1] < year:
            continue
        row_sum = teamsvecs[values][yid[i-1][0]:yid[i][0]].sum(axis=0)
        value_year[i-1] = row_sum
    value_year[-1] = teamsvecs[values][yid[-1][0]:].sum(axis=0)
    year_value = value_year.T
    sum_value_in_all_years = year_value.sum(axis=1)
    values_bigger_than = sum_value_in_all_years > 300
    year_value_hm = year_value[values_bigger_than.nonzero()[0]]
    
    # plt.xticks(color='w')
    # plt.yticks(color='w')
    plt.pcolormesh(year_value_hm, cmap='gray_r')
    plt.colorbar()
    plt.savefig(f'{dataset_path}/{dname}_{values}_heatmap_after_{year}_gray_r.pdf', dpi=100, bbox_inches='tight')
    plt.show()

def hmap(dataset_path, values):
    dname = dataset_path.split('/')[-2]
    with open(f'{dataset_path}/teamsvecs.pkl', 'rb') as infile:
        teamsvecs = pickle.load(infile)
    with open(f'{dataset_path}/indexes.pkl', 'rb') as infile:
        indexes = pickle.load(infile)
    year_idx = indexes['i2y']
    value_year = np.empty((len(year_idx), teamsvecs[values].shape[1]))
    for i in range(1, len(year_idx)):
        row_sum = teamsvecs[values][year_idx[i-1][0]:year_idx[i][0]].sum(axis=0)
        value_year[i-1] = row_sum
    value_year[-1] = teamsvecs[values][year_idx[-1][0]:].sum(axis=0)
    year_value = value_year.T
    sum_value_in_all_years = year_value.sum(axis=1)
    values_bigger_than = sum_value_in_all_years > 1000
    year_value_hm = year_value[values_bigger_than.nonzero()[0]]
    
    # plt.xticks(color='w')
    # plt.yticks(color='w')
    plt.pcolormesh(year_value_hm, cmap='gray_r')
    plt.colorbar()
    plt.savefig(f'{dataset_path}/{dname}_{values}_heatmap_gray_r.pdf', dpi=100, bbox_inches='tight')
    plt.show()

# for dataset in datasets:
    # hmap_after_year(dataset, 2005)
    # hmap(dataset)
