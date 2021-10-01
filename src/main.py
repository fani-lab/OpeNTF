import sys,os
import argparse

from cmn.publication import Publication
from dal.data_utils import *
import dnn
import nmt
import sgns

sys.path.extend(['../cmn'])
def run(datapath, domain, filter, topn, min_team_size, min_team, model, ncores, output, cmd):
    preprocessed_path = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
    filtered_path = f'./../data/filtered/{os.path.split(datapath)[-1]}'

    if domain == 'dblp':
        id_vecs, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i = Publication.generate_sparse_vectors(datapath, filter, preprocessed_path, filtered_path, min_team_size, min_team, topn, ncores)

    splits = create_evaluation_splits(len(t2i), 5)

    if model == 'dnn':
        dnn.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'{output}{os.path.split(datapath)[-1]}/fnn', cmd=['train', 'eval', 'test'])  # ['train', 'test', 'eval']

    # nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

    if model == 'sgns':
        sgns.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/{output}/sgns', cmd=['train', 'eval', 'test'])   # ['test', 'plot', 'eval']


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', type=str, required=True, help='The dataset path; required; (example: ./../data/raw/toy.json.json)')
    dataset.add_argument('-domain', type=str, required=True, choices=['dblp', 'imdb', 'patent'], help='The dataset path; required; (example: dblp)')
    dataset.add_argument('-filter', type=int, required=True, default=1, choices=[1, 0], help='Remove outliers? (1=True, 0=False)')
    dataset.add_argument('-topn', type=int, default=None, help='The topn instances; (default: all)')
    dataset.add_argument('-min_team_size', type=int, default=3, help='Discard teams with smaller size (if filter == 1)')
    dataset.add_argument('-min_team', type=int, default=3, help='Discard members with fewer teams (if filter == 1)')
    dataset.add_argument('-ncores', type=int, default=4, help='Number of cores; (default: 4 [-1 for all cores])')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', type=str, required=True, choices=['dnn', 'sgns', 'nmt'], help='The model name (example: dnn)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: ./../output/)')


# python -u main.py -data=./../data/raw/toy.json.json -domain=dblp -topn=10000 -model=dnn -ncores=4 2>&1 | tee ./../output/toy.json.log &
# python -u main.py -data=./../data/raw/dblp.v12.json.json.json -domain=dblp -topn=10000 -model=dnn -ncores=4 2>&1 | tee ./../output/dblp.log &
global ncores
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Team Formation')
    addargs(parser)
    args = parser.parse_args()

    run(datapath=args.data,
        domain=args.domain,
        filter=args.filter,
        topn=args.topn,
        min_team_size=args.min_team_size,
        min_team=args.min_team,
        model=args.model,
        ncores=args.ncores,
        output=args.output,
        cmd=['train', 'eval', 'test'])  # cmd=['train', 'test', 'eval']

