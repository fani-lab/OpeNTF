import pickle
from scipy.sparse import lil_matrix, coo_matrix
def reformat_pickle(path):
    with open(f'{path}/sparse.pkl', 'rb') as infile:
        print("Loading the stats pickle ...")
        data = pickle.load(infile)
    len_dataset = data[0].get_shape()[0]
    ids = list(range(len_dataset))
    skills = []
    users = []
    dataset = []
    for i in range(len_dataset):
        skills.append(data[0].getrow(i).tocoo())
        users.append(data[1].getrow(i).tocoo())
    for i in range(len_dataset):
        dataset.append([ids[i], skills[i], users[i]])
    with open(f'{path}/radin_sparse.pkl', 'wb') as outfile:
        pickle.dump(dataset, outfile)

reformat_pickle('../../data/preprocessed/toy')