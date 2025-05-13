import pickle

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from scipy.sparse import lil_matrix
from cmn.sparse_sgd import SparseSGD

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



