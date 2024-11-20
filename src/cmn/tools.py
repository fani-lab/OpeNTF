import pickle

import numpy as np
import pandas as pd
import scipy.sparse
import copy
import torch
from json import JSONEncoder
from scipy.sparse import lil_matrix
from cmn.sparse_sgd import SparseSGD


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


def get_class_data_params_n_optimizer(nr_classes, lr, device):
    class_parameters = torch.tensor(np.ones(nr_classes) * np.log(1.0),
                                    dtype=torch.float32,
                                    requires_grad=True,
                                    device=device)
    optimizer_class_param = SparseSGD([class_parameters],
                                      lr=lr,
                                      momentum=0.9,
                                      skip_update_zero_grad=True)
    print('Initialized class_parameters with: {}'.format(1.0))
    print('optimizer_class_param:')
    print(optimizer_class_param)

    return class_parameters, optimizer_class_param


def adjust_learning_rate(model_initial_lr, optimizer, gamma, step):
    lr = model_initial_lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def apply_weight_decay_data_parameters(loss, class_parameter_minibatch, weight_decay):
    loss = loss + 0.5 * weight_decay * (class_parameter_minibatch ** 2).sum()

    return loss


def generate_popular_and_nonpopular(vecs, input_path):
    """

    Parameters
    ----------
    vecs: the lil matrix representation of the dataset
    input_path: the path of the dataset

    Returns: None,

    This function Saves two datasets{
                                    popular_inst: teams that have at least one popular expert
                                    non_popular_inst: teams that do not have any popular experts
                                }
    -------
    This function needs a **popularity matrix** that can be generated from Adila submodule.
    The objective of this function is to divide teams into popular and non-popular subsets.
    """
    try:
        popularity = pd.read_csv(input_path + '/popularity.csv', index_col='memberidx')
    except FileNotFoundError:
        print(f"To start, copy and paste popularity file in {input_path}.")
        return FileNotFoundError
    popularity = popularity.to_numpy().squeeze()

    popular_vecs = {
        'id': lil_matrix((0, vecs['id'].shape[1]), dtype=vecs['id'].dtype),
        'skill': lil_matrix((0, vecs['skill'].shape[1]), dtype=vecs['skill'].dtype),
        'member': lil_matrix((0, vecs['member'].shape[1]), dtype=vecs['member'].dtype)
    }
    non_popular_vecs = {
        'id': lil_matrix((0, vecs['id'].shape[1]), dtype=vecs['id'].dtype),
        'skill': lil_matrix((0, vecs['skill'].shape[1]), dtype=vecs['skill'].dtype),
        'member': lil_matrix((0, vecs['member'].shape[1]), dtype=vecs['member'].dtype)
    }
    length = len(vecs['member'].rows.tolist())
    for i, row in enumerate(vecs['member'].rows.tolist()):
        popular = False
        print(f'{i} / {length}')
        for expert in row:
            try:
                if popularity[expert]:
                    popular = True
                    break
            except IndexError:
                continue
        if popular:
            popular_vecs['id'] = scipy.sparse.vstack([popular_vecs['id'], vecs['id'][i]]).tolil()
            popular_vecs['skill'] = scipy.sparse.vstack([popular_vecs['skill'], vecs['skill'][i]]).tolil()
            popular_vecs['member'] = scipy.sparse.vstack([popular_vecs['member'], vecs['member'][i]]).tolil()
        else:
            non_popular_vecs['id'] = scipy.sparse.vstack([non_popular_vecs['id'], vecs['id'][i]]).tolil()
            non_popular_vecs['skill'] = scipy.sparse.vstack([non_popular_vecs['skill'], vecs['skill'][i]]).tolil()
            non_popular_vecs['member'] = scipy.sparse.vstack([non_popular_vecs['member'], vecs['member'][i]]).tolil()

    with open(input_path + '/popular_inst.pkl', 'wb') as file:
        pickle.dump(popular_vecs, file)
        print("popular vecs are saved!")
    with open(input_path + '/non_popular_inst.pkl', 'wb') as file:
        pickle.dump(non_popular_vecs, file)
        print("nonpopular vecs are saved!")


def popular_nonpopular_ratio(vecs, input_path, ratio=0):
    """

    Parameters
    ----------
    vecs: the lil matrix representation of the dataset
    input_path: the path of the dataset
    ratio: {
        0 -> Only popular experts will remain in vecs['member']
        1 -> the proportion of popular and non-popular experts are the same
        >1 -> currently returns only the input
    }

    Returns: an edited version of vecs
    -------
    place these 3 lines before sreate_evaluation_splits():
    # with open(data_list[0] + '/popular_inst.pkl', 'rb') as file:
        #     vecs = pickle.load(file)
    # vecs = popular_nonpopular_ratio(vecs, data_list[0], ratio=0)

    """
    if ratio > 1:
        return vecs
    try:
        popularity = pd.read_csv(input_path + '/popularity.csv', index_col='memberidx')
    except FileNotFoundError:
        print(f"To start, copy and paste popularity file in {input_path}.")
        return FileNotFoundError
    popularity = popularity.to_numpy().squeeze()
    popular_count = popularity[popularity == True].shape[0]
    nonpopular_count = popularity[popularity == False].shape[0]
    print(f"number of popular experts: {popular_count} -> {popular_count / popularity.shape[0] * 100} % \nNumber of non-popular experts {nonpopular_count} -> {nonpopular_count / popularity.shape[0] * 100} % ")
    if ratio == 0:
        return {
            'id': vecs['id'],
            'skill': vecs['skill'],
            'member': vecs['member'][:, popularity],
        }

# teamsvecs = {}
# 1 110 0110
# 2 110 1110
# ------------
# 3 011 0111
# 4 011 0110
# ------------
# 5 111 1110
# teamsvecs['id'] = scipy.sparse.lil_matrix([[1],[2],[3],[4],[5]])
# teamsvecs['skill'] = scipy.sparse.lil_matrix([[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]])
# teamsvecs['member'] = scipy.sparse.lil_matrix([[0,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,0],[1,1,1,0]])
#
# new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=True)
# print(new_teamsvecs['id'].todense())# <= [[1], [3], [5]]
# print(new_teamsvecs['skill'].todense())# <= [[1, 1, 0], [0, 1, 1], [1, 1, 1]]
# print(new_teamsvecs['member'].todense())# <= [[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]
#
# new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=False)
# print(new_teamsvecs['id'].todense())# <= [[1],[2],[3],[4],[5]]
# print(new_teamsvecs['skill'].todense())# <= [[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]]
# print(new_teamsvecs['member'].todense())# <= [[1,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,1],[1,1,1,0]]

