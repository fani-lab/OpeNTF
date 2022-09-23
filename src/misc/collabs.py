import pickle
from tqdm import tqdm
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci
from itertools import combinations
from math import comb


# Returns the pairwise collaborations
# teams_members: sparse matrix of teams['member']
def get2WayCollabs(teams_members):
    return teams_members.transpose() @ teams_members

def getnWayCollabs(teams_members, n, threshold=1):
    rowIndexes = list(range(0, teams_members.shape[1]))

    # Will record the count for each combination
    # The index for count aligns with it's respective combination
    teams_members = teams_members.transpose()
    collabs = []
    for testCase in tqdm(combinations(rowIndexes, n), total=comb(len(rowIndexes), n)):
        # dotProduct = teams_members.getrow(testCase[0])
        # for i in range(1, n): dotProduct = dotProduct.multiply(teams_members.getrow(testCase[i]))
        dotProduct = (teams_members.getrow(testCase[0]).toarray())[0]
        for i in range(1, n): dotProduct = dotProduct * (teams_members.getrow(testCase[i]).toarray())[0]
        finalDotProduct = np.sum(dotProduct)
        # Do not append if there are less thank threshold collaborations
        if(finalDotProduct > threshold): collabs.append([testCase, finalDotProduct])

    return collabs

def getTopK_nWays(teams_members, nway, k=10, threshold=1):
    n_way_collabs = getnWayCollabs(teams_members, nway, threshold)
    n_way_collabs.sort(key=lambda x: x[1], reverse=True)
    return n_way_collabs[0:k]

# Plots Results of top-k into a Histogram
# result: an array with top-K
def plotTopK_nWays(result, names=None, savefig=None):
    if len(result) < 1:
        print('no data to plot')
        return
    data = []  # The number of repetitions
    indices = []  # The Coordinates
    for team in result:
        indices.append(team[0])
        data.append(team[1])

    k = len(data)
    fig, ax = plt.subplots()
    # Form X-Axis Categories:
    xAxis = []
    for i in range(0, k): xAxis.append(f"({','.join([names[j] for j in indices[i]])})") if names else xAxis.append(
        str(indices[i]))

    ax.bar(xAxis, height=data)
    ax.set_title(f"Top-{str(k)} for {len(indices[0])}-Way Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Collabs")
    plt.xticks(rotation=90)
    if savefig: plt.savefig(savefig, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def main():
    # Test teams: (0,1), (2,3), (0,1,3), (0,1,3), (0,2,3)
    # Test Matrix: rows=teams, columns=members
    A = sci.coo_matrix([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]])
    names = None
    # 2 way results: (0,1)=3, (0,3)=3, (1,3)=2, (2,3)=2, (0,2)=1
    # 3 way results: (0,1,3)=2, (0,2,3)=1
    # 4 way results: None

    # test performance
    # p = 0.9
    # N = 200
    # A = sci.coo_matrix(np.random.choice(a=[False, True], size=(N, N), p=[p, 1-p]))
    # getTopK_nWays(A, nway=2, k=10, threshold=5)

    with open('../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl', 'rb') as f: matrix=pickle.load(f)
    A = matrix['member']

    with open('../../data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl', 'rb') as f: indexes=pickle.load(f)
    names = indexes['i2c']

    # #entire dataset
    # with open('../../data/preprocessed/dblp/dblp.v12.json/teamsvecs.pkl', 'rb') as f: matrix=pickle.load(f)
    # A = matrix['member']
    # with open('../../data/preprocessed/dblp/dblp.v12.json/indexes.pkl', 'rb') as f: indexes=pickle.load(f)
    # names = indexes['i2c']

    #filtered
    with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/teamsvecs.pkl', 'rb') as f: matrix=pickle.load(f)
    A = matrix['member']
    with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/indexes.pkl', 'rb') as f: indexes=pickle.load(f)
    names = indexes['i2c']

    # print(A.asformat("array"))
    # plotTopK_nWays(getTopK_nWays(A, nway=2, k=10, threshold=0), names=names)
    # plotTopK_nWays(getTopK_nWays(A, nway=3, k=10, threshold=0), names=names)
    # plotTopK_nWays(getTopK_nWays(A, nway=4, k=10), names=names)

    # data = getTopK_nWays(A, nway=2, k=10, threshold=5)
    # with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs-2-top10.pkl', 'wb') as f: pickle.dump(data,f)
    # plotTopK_nWays(data, names=names, savefig="./data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs-2-top10.png")

    data = getTopK_nWays(A, nway=2, k=1000, threshold=5)#100%|██████████| 101011791/101011791 [4:32:39<00:00, 6174.42it/s]
    with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs-2-top1000.pkl', 'wb') as f: pickle.dump(data,f)
    plotTopK_nWays(data, names=names, savefig="./data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs-2-top1000.png")

    # #not doable! ((478526524564/3500)/3600)/24 = 1582 days
    # data = getTopK_nWays(A, nway=3, k=10, threshold=5)#0%|          | 789041/478526524564 [03:19<36326:36:56, 3659.13it/s]
    # with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs-3.pkl', 'wb') as f: pickle.dump(data, f)

    # with multiprocessing.Pool() as p:
    #     func = partial(getTopK_nWays, A)
    #     data = p.map(func, [2,3,4,5])
    # with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs.pkl', 'wb') as f: pickle.dump(data, f)

if __name__ == '__main__':
    main()

