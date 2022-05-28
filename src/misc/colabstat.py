import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sci

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



# Main Function for Testing:
def main():  
    # Test Matrix:
    A = sci.coo_matrix([[1,1,0,0], [0,0,1,1], [0,1,1,0], [1,1,1,1], [0,0,1,1]])

    outMat = getPairwiseCollabs(A)
    topFour = getTopK(outMat, 4)

    plotTopK(topFour)


# Uncomment to run main:
# main()