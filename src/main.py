import sys,os
import argparse

from cmn.publication import Publication
from dal.data_utils import *
import dnn
import nmt
import sgns

sys.path.extend(['../cmn'])
def run(datapath, domain, topn, ncores, output, cmd):

    if domain == 'dblp':
        id_vecs, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i = Publication.generate_sparse_vectors(datapath, f'./../data/preprocessed/{os.path.split(datapath)[-1]}', topn, ncores)

    # splits = create_evaluation_splits(len(t2i), 2)

    # dnn.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/{output}/fnn', cmd=['train', 'eval', 'test'])  # ['train', 'test', 'eval']

    # nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])
    #
    # sgns.main(splits, skill_vecs, member_vecs, i2m, m2i, i2s, s2i, output=f'../output/{output}/sgns', cmd=['train', 'eval', 'test'])   # ['test', 'plot', 'eval']


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', type=str, required=True, help='The dataset path; required; (example: ./../data/raw/toy.json)')
    dataset.add_argument('-domain', type=str, required=True, choices=['dblp', 'imdb', 'patent'], help='The dataset path; required; (example: dblp)')
    dataset.add_argument('-topn', type=int, default=None, help='The topn instances; (default: all)')
    dataset.add_argument('-ncores', type=int, default=4, help='Number of cores; (default: 4 [-1 for all cores])')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', type=str, required=True, choices=['dnn', 'nmt'], help='The model name (example: dnn)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: ./../output/)')


# python -u main.py -data=./../data/raw/toy.json -domain=dblp -topn=10000 -model=dnn -ncore=4 2>&1 | tee ./../output/toy.log &
# python -u main.py -data=./../data/raw/dblp.v12.json -domain=dblp -topn=10000 -model=dnn -ncore=4 2>&1 | tee ./../output/dblp.log &
global ncores
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Team Formation')
    addargs(parser)
    args = parser.parse_args()

    run(datapath=args.data,
        domain=args.domain,
        topn=args.topn,
        ncores=args.ncores,
        output=args.output,
        cmd=['train', 'eval', 'test'])  # cmd=['train', 'test', 'eval']

