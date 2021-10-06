import sys,os
import argparse

import param
from cmn.publication import Publication
from cmn.movie import Movie 
from dal.data_utils import *
import dnn
import nmt
import sgns

sys.path.extend(['../cmn'])

def run(datapath, domain, filter, model, output, settings):
    if domain == 'dblp':
        vecs, indexes = Publication.generate_sparse_vectors(datapath, f'./../data/preprocessed/dblp/{os.path.split(datapath)[-1]}', filter, settings['data'])

    if domain == 'imdb':
        vecs, indexes = Movie.generate_sparse_vectors(datapath, f'./../data/preprocessed/imdb/{os.path.split(datapath)[-1]}', filter, settings['data'])

    splits = create_evaluation_splits(len(indexes['t2i']), settings['model']['splits'])

    if model == 'dnn':
        dnn.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}/dnn', settings['model']['baseline']['dnn'], settings['model']['cmd'])

    # nmt.main(splits, input_data, output_data, cmd=['train', 'test', 'eval'])

    if model == 'sgns':
        sgns.main(splits, vecs['skill'], vecs['member'], indexes, f'../output/{output}/sgns', settings['model'])


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', type=str, required=True, help='The dataset path; required; (example: ./../data/raw/toy.json)')
    dataset.add_argument('-domain', type=str, required=True, choices=['dblp', 'imdb', 'patent'], help='The dataset path; required; (example: dblp)')
    dataset.add_argument('-filter', type=int, default=1, choices=[1, 0], help='Remove outliers? (1=True (default), 0=False)')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', type=str, required=True, choices=['dnn', 'sgns', 'nmt'], help='The model name (example: dnn)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: ./../output/)')

# python -u main.py -data=./../data/raw/toy.json -domain=dblp -model=dnn  2>&1 | tee ./../output/toy.log &
# python -u main.py -data=./../data/raw/dblp.v12.json -domain=dblp -model=dnn 2>&1 | tee ./../output/dblp.log &
# python -u main.py -data=./../data/raw/toy.title.basics.tsv -domain=imdb -model=dnn -filter=0
# python -u main.py -data=./../data/raw/title.basics.tsv -domain=imdb -model=dnn -filter=0

global ncores
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Team Formation')
    addargs(parser)
    args = parser.parse_args()

    run(datapath=args.data,
        domain=args.domain,
        filter=args.filter,
        model=args.model,
        output=args.output,
        settings=param.settings)

