import pickle
from scipy.sparse import lil_matrix, coo_matrix
def reformat_pickle(path):
    with open(f'{path}/teamsvecs.pkl', 'rb') as infile:
        print("Loading the teamsvec pickle ...")
        data = pickle.load(infile)
    len_dataset = data['id'].get_shape()[0]
    ids = []
    skills = []
    users = []
    dataset = []
    for i in range(len_dataset):
        ids.append(data['id'].getrow(i).tocoo())
        skills.append(data['skill'].getrow(i).tocoo())
        users.append(data['member'].getrow(i).tocoo())
    for i in range(len_dataset):
        dataset.append([ids[i], skills[i], users[i]])
    with open(f'{path}/radin_sparse_dblp_filtered3.pkl', 'wb') as outfile:
        pickle.dump(dataset, outfile)

reformat_pickle('../data/preprocessed/dblp/dblp.v12.json.filtered.mt5.ts3')