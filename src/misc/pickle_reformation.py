import pickle
from scipy.sparse import lil_matrix, coo_matrix
def reformat_pickle_with_path(path):
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

def reformat_pickle(vecs, output, filter, settings):
    output += f'.filtered.mt{settings["filter"]["min_nteam"]}.ts{settings["filter"]["min_team_size"]}' if filter else ""
    path = output
    try:
        with open(f'{path}/teamsvecs_for_emb.pkl', 'rb') as infile:
            print("Loading the teamsvec pickle ...")
            dataset = pickle.load(infile)
            return dataset, path
    except:

        len_dataset = vecs['id'].get_shape()[0]
        # print("len_dataset:", vecs['member'].get_shape())
        ids = []
        skills = []
        users = []
        dataset = []
        for i in range(len_dataset):
            ids.append(vecs['id'].getrow(i).tocoo())
            skills.append(vecs['skill'].getrow(i).tocoo())
            users.append(vecs['member'].getrow(i).tocoo())
        for i in range(len_dataset):
            dataset.append([ids[i], skills[i], users[i]])
        with open(f'{path}/teamsvecs_for_emb.pkl', 'wb') as outfile:
            pickle.dump(dataset, outfile)
        return dataset, path
