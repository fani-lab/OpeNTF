import numpy as np
import scipy.sparse
import copy
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def merge_teams_by_skills(teamsvecs, inplace=False, distinct=False):
    vecs = teamsvecs if inplace else copy.deepcopy(teamsvecs)
    merge_list = {}

    # in the following loop rows that have similar skills are founded
    for i in range(len(vecs['skill'].rows)):
        merge_list[f'{i}'] = set()
        for j in range(i + 1, len(vecs['skill'].rows)):
            if vecs['skill'].rows[i] == vecs['skill'].rows[j]: merge_list[f'{i}'].add(j)
        if len(merge_list[f'{i}']) < 1: del merge_list[f'{i}']

    delete_set = set()
    for key in merge_list.keys():
        for item in merge_list[key]: delete_set.add(item)

    for item in delete_set:
        try:
            del merge_list[f'{item}']
        except KeyError:
            pass

    del_list = []
    for key_ in merge_list.keys():
        for value_ in merge_list[key_]:
            del_list.append(value_)
            vec1 = vecs['member'].getrow(int(key_))
            vec2 = vecs['member'].getrow(value_)
            result = np.add(vec1, vec2)
            result[result != 0] = 1
            vecs['member'][int(key_), :] = scipy.sparse.lil_matrix(result)
            vecs['member'][int(value_), :] = scipy.sparse.lil_matrix(result)
    if distinct:
        vecs['id'] = scipy.sparse.lil_matrix(np.delete(vecs['id'].toarray(), del_list, axis=0))
        vecs['skill'] = scipy.sparse.lil_matrix(np.delete(vecs['skill'].toarray(), del_list, axis=0))
        vecs['member'] = scipy.sparse.lil_matrix(np.delete(vecs['member'].toarray(), del_list, axis=0))
    return vecs

teamsvecs = {}
# 1 110 0110
# 2 110 1110
# ------------
# 3 011 0111
# 4 011 0110
# ------------
# 5 111 1110
teamsvecs['id'] = scipy.sparse.lil_matrix([[1],[2],[3],[4],[5]])
teamsvecs['skill'] = scipy.sparse.lil_matrix([[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]])
teamsvecs['member'] = scipy.sparse.lil_matrix([[0,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,0],[1,1,1,0]])

new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=True)
print(new_teamsvecs['id'].todense())# <= [[1], [3], [5]]
print(new_teamsvecs['skill'].todense())# <= [[1, 1, 0], [0, 1, 1], [1, 1, 1]]
print(new_teamsvecs['member'].todense())# <= [[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]

new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=False)
print(new_teamsvecs['id'].todense())# <= [[1],[2],[3],[4],[5]]
print(new_teamsvecs['skill'].todense())# <= [[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]]
print(new_teamsvecs['member'].todense())# <= [[1,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,1],[1,1,1,0]]

