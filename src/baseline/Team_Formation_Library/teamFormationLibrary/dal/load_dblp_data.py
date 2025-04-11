from os import path
from TFL import *
import pickle

def nn_t2v_dataset_generator(model, dataset, output_file_path, mode='user'):
    """Generates the T2V dataset
    Generates the T2V dataset from the embeddings and saves the
    file to a provided location
    Parameters
    ----------
    model : Embedding
        The embedding model generated previously
    dataset : pkl format
        The preprocessed dataset
    output_file_path : string
        The local path where the T2V datasets will be saved
    mode : string
        Whether to generate user or skill dataset
    """
    t2v_dataset = []
    counter = 1
    for record in dataset:
        id = record[0]
        if mode.lower() == 'user':
            try:
                skill_vec = record[1].todense()
                team_vec = model.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'skill':
            try:
                skill_vec = model.get_team_vec(id)
                team_vec = record[2].todense()
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
        elif mode.lower() == 'full':
            try:
                model_skill = model['skill']
                model_user = model['user']
                skill_vec = model_skill.get_team_vec(id)
                team_vec = model_user.get_team_vec(id)
                t2v_dataset.append([id, skill_vec, team_vec])
                print('Record #{} | File #{} appended to dataset.'.format(counter, id))
                counter += 1
            except:
                print('Cannot add record with id {}'.format(id))
    with open(output_file_path, 'wb') as f:
        pickle.dump(t2v_dataset, f)


def preprocessed_dataset_exist(file_path='dataset/dblp_preprocessed_dataset.pkl'):
    """Determines whether the preprocessed dataset
    exists in the provided location
    Parameters
    ----------
    file_path : string
        The local path where the preprocessed dataset is located
    """
    if path.exists(file_path):
        return True
    return False


def load_preprocessed_dataset(file_path='dataset/dblp_preprocessed_dataset.pkl'): #'dataset/dblp_preprocessed_dataset.pkl'
    """Load the preprocessed dataset from the local
    path provided by the user
    Parameters
    ----------
    file_path : string
        The local path where the preprocessed dataset is located
    """
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
