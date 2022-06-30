import pickle
import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci
from itertools import combinations


# Returns the pairwise collaborations
# teams_members: sparse matrix of teams['member']
def get2WayCollabs(teams_members):
    return teams_members.transpose() @ teams_members


def getTopK_nWays(teams_members, nway, k=10):
    n_way_collabs = getnWayCollabs(teams_members, nway)
    for i in range(0, len(n_way_collabs) - 1):
        max = i
        for j in range(i + 1, len(n_way_collabs)):
            if (n_way_collabs[j][1] > n_way_collabs[max][1]):
                max = j
        n_way_collabs[max], n_way_collabs[i] = n_way_collabs[i], n_way_collabs[max]
    return n_way_collabs[0:k]


# Plots Results of top-k into a Histogram
# result: an array with top-K
def plotTopK_nWays(result, names=None):
    if len(result) < 1:
        print('no data to plot')
        return
    data = []  # The number of repetitions
    indices = []  # The Coordinates
    for team in result:
        indices.append(team[0])
        data.append(team[1])

    k = len(data)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])

    # Form X-Axis Categories:
    xAxis = []
    for i in range(0, k): xAxis.append(f"({','.join([names[j] for j in indices[i]])})") if names else xAxis.append(
        str(indices[i]))

    ax.bar(xAxis, height=data)
    ax.set_title(f"Top-{str(k)} for {len(indices[0])}-Way Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Collabs")
    plt.xticks(rotation=90)
    plt.show()


def getnWayCollabs(teams_members, n):
    rowIndexes = []
    for i in range(0, teams_members.shape[1]): rowIndexes.append(i)

    # comb is a list of all the possible combinations (if there are 4 rows) = [0,1,2], [0,1,3], ...
    comb = combinations(rowIndexes, n)

    # Will record the count for each combination
    # The index for count aligns with it's respective combination
    collabs = []
    for testCase in list(comb):
        dotProduct = (teams_members.transpose().getrow(testCase[0]).toarray())[0]
        for i in range(1, n):
            dotProduct = dotProduct * (teams_members.transpose().getrow(testCase[i]).toarray())[0]
        collabs.append([testCase, np.sum(dotProduct)])

    return collabs

def main():
    # Test teams: (0,1), (2,3), (0,1,3), (0,1,3), (0,2,3)
    # Test Matrix: rows=teams, columns=members
    A = sci.coo_matrix([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]])
    names = None
    # 2 way results: (0,1)=3, (0,3)=3, (1,3)=2, (2,3)=2, (0,2)=1
    # 3 way results: (0,1,3)=2, (0,2,3)=1
    # 4 way results: None

    # with open('../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl', 'rb') as f: matrix=pickle.load(f)
    # A = matrix['member']
    #
    # with open('../../data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl', 'rb') as f: indexes=pickle.load(f)
    # names = indexes['i2c']

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
    # plotTopK_nWays(getTopK_nWays(A, nway=2, k=10), names=names)
    # plotTopK_nWays(getTopK_nWays(A, nway=3, k=10), names=names)
    # plotTopK_nWays(getTopK_nWays(A, nway=4, k=10), names=names)


    with multiprocessing.Pool() as p:
        func = partial(getTopK_nWays, A)
        data = p.map(func, [2,3,4,5])
    with open('../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/collabs.pkl', 'wb') as f: pickle.dump(data, f)

if __name__ == '__main__':
    main()

