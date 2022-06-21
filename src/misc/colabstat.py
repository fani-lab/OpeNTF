
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci

from itertools import combinations

# Returns the pairwise collaborations
# teams_members: sparse matrix of teams['member']
def get2WayCollabs(teams_members):
    return teams_members.transpose() @ teams_members



def getTopK_nWays(n_way_collabs, k):
    for i in range(0, len(n_way_collabs) - 1):
        max = i
        for j in range(i+1, len(n_way_collabs)):
            if(n_way_collabs[j][1] > n_way_collabs[max][1]):
                max = j
        n_way_collabs[max], n_way_collabs[i] = n_way_collabs[i], n_way_collabs[max]
    return n_way_collabs[0:k]
        

# Plots Results of top-k into a Histogram
# result: an array with top-K
def plotTopK_nWays(result, names=None):
    if len(result) < 1:
        print('no data to plot')
        return
    data = [] # The number of repetitions
    indices = [] # The Coordinates
    for team in result:
        indices.append(team[0])
        data.append(team[1])

    k = len(data)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.7,0.7])
    
    # Form X-Axis Categories:
    xAxis = []
    for i in range(0,k): xAxis.append(f"({','.join([names[j] for j in indices[i]])})") if names else xAxis.append(str(indices[i]))
    
    ax.bar(xAxis,height=data)
    ax.set_title(f"Top-{str(k)} for {len(indices[0])}-Way Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Collabs")
    plt.xticks(rotation=90)
    plt.show()





def getnWayCollabs(team_member, n):
    rowIndexes = []
    for i in range(0, teams_member.shape[1]): rowIndexes.append(i)

    # comb is a list of all the possible combinations (if there are 4 rows) = [0,1,2], [0,1,3], ...
    comb = combinations(rowIndexes, n) 

    # Will record the count for each combination
    # The index for count aligns with it's respective combination
    collabs = []
    for testCase in list(comb):
        dotProduct = (team_member.transpose().getrow(testCase[0]).toarray())[0]
        for i in range(1, n):
            dotProduct = dotProduct * (team_member.transpose().getrow(testCase[i]).toarray())[0]
        collabs.append([testCase, np.sum(dotProduct)])
    
    return collabs


def nwayCollabs(teams_members, nway, topk, names):
    m = getnWayCollabs(teams_members, nway)
    plotTopK_nWays(getTopK_nWays(m, topk), names)



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

    print(A.asformat("array"))
    nwayCollabs(A, nway=2, topk=5, names=names)
    nwayCollabs(A, nway=3, topk=5, names=names)
    nwayCollabs(A, nway=4, topk=5, names=names)

main()



''' 
Old getnWayCollabs() functions:



# Format: [coordinate, number]
def getnWayCollabs(teams_member, n):
    rowIndexes = []
    for i in range(0, teams_member.shape[1]): rowIndexes.append(i)

    # comb is a list of all the possible combinations (if there are 4 rows) = [0,1,2], [0,1,3], ...
    comb = combinations(rowIndexes, n) 

    # Will record the count for each combination
    # The index for count aligns with it's respective combination
    collabs = []
    for testCase in list(comb):
        tempCount = 0
        for i in range(0, teams_member.shape[0]):
            row = (teams_member.getrow(i).asformat("array"))[0]
            for j in range(0, n):
                if(row[testCase[j]] != 1):
                    tempCount -= 1
                    break
            tempCount += 1

        if tempCount != 0:
            collabs.append([testCase, tempCount])

    return collabs

# Plots Results of top-k into a Histogram
# result: a coo_matrix of the top-k
def plotTopK_2Ways(result):
    # indices[0] for x coordinate
    # indices[1] for y coordinate
    indices = result.nonzero()
    data = result.data
    k = len(data) # Gets the k value

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.7,0.7])
    
    # Form X-Axis Categories:
    xAxis = []
    for i in range(0,k): xAxis.append("(" + str(indices[0][i]) + "," + str(indices[1][i]) + ")")
    
    ax.bar(xAxis,height=data)
    ax.set_title("Top " + str(k) + " 2-Way Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Colabs")
    plt.show()


# Returns the top-k in coo_matrix format
# two_way_collabs: matrix of pairwise collaborations
# k: top-k
def getTopK_2Ways(two_way_collabs, k):
    result = [0] * k
    upperTri = sci.triu(two_way_collabs)
    upperTri.setdiag(0)

    indexes = sci.find(upperTri)

    # Sorts the List of Elements in Upper Triangular
    for i in range(0, len(indexes[0]) -1):
        max = i
        for j in range(i+1, len(indexes[0])):
            if(indexes[2][j] > indexes[2][max]): max = j

        # Swap the "i" (row) coordinate:
        indexes[0][max], indexes[0][i] = indexes[0][i], indexes[0][max]

        # Swap the "j" (column) coordinate:
        indexes[1][max], indexes[1][i] = indexes[1][i], indexes[1][max]

        # Swap the values
        indexes[2][max], indexes[2][i] = indexes[2][i], indexes[2][max]
    
    rows, columns, data = indexes[0][0:k], indexes[1][0:k], indexes[2][0:k]
    return sci.coo_matrix((data, (rows, columns)),shape=upperTri.get_shape())


'''
