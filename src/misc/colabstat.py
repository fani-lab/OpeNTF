import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci

from itertools import combinations

# Returns the pairwise collaborations
# m_matrix: sparse matrix of teams
def getPairwiseCollabs(m_matrix):
    return m_matrix.transpose() @ m_matrix

# Returns the top-k in coo_matrix format
# m_matrix: matrix of pairwise collaborations
# k: top-k
def getTopK(m_matrix, k):
    result = [0] * k
    upperTri = sci.triu(m_matrix)

    upperTri.setdiag(0)

    indexes = sci.find(upperTri)

    # Sorts the List of Elements in Upper Triangular
    for i in range(0, len(indexes[0]) -1):
        max = i
        for j in range(i+1, len(indexes[0])):
            if(indexes[2][j] > indexes[2][max]):
                max = j
        # Swap the "i" (row) coordinate:
        temp = indexes[0][i]
        indexes[0][i] = indexes[0][max]
        indexes[0][max] = temp
 
        # Swap the "j" (column) coordinate:
        temp = indexes[1][i]
        indexes[1][i] = indexes[1][max]
        indexes[1][max] = temp

        # Swap the values
        temp = indexes[2][i]
        indexes[2][i] = indexes[2][max]
        indexes[2][max] = temp
    

    rows = indexes[0][0:k]
    columns = indexes[1][0:k]
    data = indexes[2][0:k]


    return sci.coo_matrix((data, (rows, columns)),shape=upperTri.get_shape())

def getTopK_nWays(collabs, k):
    for i in range(0, len(collabs) - 1):
        max = i
        for j in range(i+1, len(collabs)):
            if(collabs[j][1] > collabs[max][1]):
                max = j
        temp = collabs[i]
        collabs[i] = collabs[max]
        collabs[max] = temp
    print(collabs)
    return collabs[0:k]
        


# Plots Results of top-k into a Histogram
# result: a coo_matrix of the top-k
def plotTopK(result):
    # indices[0] for x coordinate
    # indices[1] for y coordinate
    indices = result.nonzero()

    data = result.data
    k = len(data) # Gets the k value

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.7,0.7])
    
    # Form X-Axis Categories:
    xAxis = []
    for i in range(0,k):
        xAxis.append("(" + str(indices[0][i]) + "," + str(indices[1][i]) + ")")
    
    # Generates the Graph:
    ax.bar(xAxis,height=data)
    ax.set_title("Top " + str(k) + " Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Pairs in Order of Top-" + str(k))

    plt.show()

# Plots Results of top-k into a Histogram
# result: an array with top-K
def plotTopK_nWays(result):
    data = [] # The number of repetitions
    indices = [] # The Coordinates
    for team in result:
        indices.append(team[0])
        data.append(team[1])

    k  = len(data)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.7,0.7])
    
    # Form X-Axis Categories:
    xAxis = []
    for i in range(0,k):
        xAxis.append(str(indices[i]))
    
    # Generates the Graph:
    ax.bar(xAxis,height=data)
    ax.set_title("Top " + str(k) + " Collaborations")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Pairs in Order of Top-" + str(k))

    plt.show()


# Format: [coordinate, number]
def nCollabs_combinations(m_matrix, n):      
    rowIndexes = []
    for i in range(0, m_matrix.shape[1]):
        rowIndexes.append(i)
    

    # comb is a list of all the possible combinations
    # comb (if there are 4 rows) = [0,1,2], [0,1,3], ...
    comb = combinations(rowIndexes, n) 
    # print(list(comb))

    # Will record the count for each combination
    # The index for count aligns with it's respective combination
    collabs = []
    for testCase in list(comb):
        tempCount = 0
        for i in range(0, m_matrix.shape[0]):
            row = (m_matrix.getrow(i).asformat("array"))[0]
            for j in range(0, n):
                if(row[testCase[j]] != 1):
                    tempCount -= 1
                    break
            tempCount += 1

        if tempCount != 0:
            collabs.append([testCase, tempCount])

    # Prints the results:
    return collabs

def nwayCollabs(m_matrix, n,k):
    if n==2:
        m = getPairwiseCollabs(m_matrix)
        topK = getTopK(m, k)
        plotTopK(topK)
    else:
        m = nCollabs_combinations(m_matrix, n)
        topK = getTopK_nWays(m, k)
        plotTopK_nWays(topK)

# Main Function for Testing:
def main():  
    # Test Matrix:
    A = sci.coo_matrix([[1,1,0,0], [0,0,1,1], [1,1,0,1], [1,1,0,1], [1,0,1,1]])

    with open('data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl', 'rb') as tfile: 
        matrix=pickle.load(tfile)

    print("---\nTESTING MATRIX:")
    print(A.asformat("array"))
    nwayCollabs(A, 3, 2)





# Uncomment to run main:
# main()

