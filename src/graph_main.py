from mdl.team2vec import graph_params

def addargs(parser):
    embedding = parser.add_argument_group('Team Embedding')
    embedding.add_argument('-teamsvecs', type=str, required=True, help='The path to the teamsvecs.pkl and indexes.pkl files; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')
    embedding.add_argument('-embmodel', type=str, required=True, help='The embedding model; e.g., w2v, n2v, ...')
    embedding.add_argument('-output', type=str, required=True, help='Output folder; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')

def run(teamsvecs_file, indexes_file, embmodel, output):
    with open(teamsvecs_file, 'rb') as teamsvecs_f, open(indexes_file, 'rb') as indexes_f:
        teamsvecs, indexes = pickle.load(teamsvecs_f), pickle.load(indexes_f)

        print('---------------------------------------')
        print(f'model = {model}')
        print(f'edge_types = {edge_types}')
        print('---------------------------------------')

        if model == 'w2v': from mdl.team2vec.wnn import Wnn; mobj = Wnn(cmd)
        elif model == 'n2v':
            from mdl.team2vec.gnn import Gnn
            from torch_geometric.nn import Node2Vec
            mobj = Gnn(Node2Vec, cmd)
        elif model == 'm2v': from mdl.team2vec.gnn import Gnn; mobj = M2V(cmd)
        mobj.run()


#python -u wnn.py -teamsvecs=../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl -embtypes=skill,member,joint -dim=100 -dbow_words=1 -output=../data/preprocessed/dblp/toy.dblp.v12.json/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team Embedding')
    addargs(parser)
    args = parser.parse_args()
    run(f'{args.teamsvecs}teamsvec.pkl', f'{args.teamsvecs}indexes.pkl', args.embmodel, args.output)
