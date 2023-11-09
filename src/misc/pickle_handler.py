import pickle
import os

# reads the pickle data from the filepath
def read_data(filepath):
    try :
        with open(filepath, 'rb') as inputfile:
            teamsvecs = pickle.load(inputfile)
            print(teamsvecs)
    except FileNotFoundError:
        print('The file was not found !')
    print()
    return teamsvecs

def write_data(filepath):
    pass

if __name__ == "__main__":
    print(f'This file handles all the pickle read and write for gnn tests')
    input_filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/teams.pkl'
    output_filepath = '../../data/preprocessed/custom/gnn_emb/teams.pkl'
    # to make sure the path to output exists or gets created
    os.makedirs(output_filepath, exist_ok=True)

    read_data(input_filepath)