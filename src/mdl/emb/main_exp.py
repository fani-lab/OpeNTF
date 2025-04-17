import params
import main
def run_entire_ds(args):
    # generate for valid combinations on entire datasets
        args.output = args.teamsvecs
        args.model = 'gnn.n2v'
        for edge_type in [('member', 'm'), ([('skill', '-', 'team'), ('member', '-', 'team')], 'stm'), ([('skill', '-', 'member')], 'sm')]:
            for dir in [True, False]:
                for dup in [None, 'mean']:
                    params.settings = {
                        'graph': {
                            'edge_types': edge_type,
                            'dir': dir,
                            'dup_edge': dup
                        }
                    }
                    main.run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', args.model, f'{args.output}/{args.model.split(".")[0]}/')


#python -u main_exp.py -teamsvecs=./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt5.ts2/ -model=gnn.n2v -output=./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt5.ts2/
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Team Embedding')
    main.addargs(parser)
    args = parser.parse_args()

    datasets = ['./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt5.ts2/',
                './../../../data/preprocessed/imdb/title.basics.tsv.filtered.mt5.ts2/',
                './../../../data/preprocessed/gith/data.csv.filtered.mt5.ts2/',
                './../../../data/preprocessed/uspt/patent.tsv.filtered.mt5.ts2/']

    # import multiprocessing as mp
    # import copy
    # argss = []
    # for ds in datasets:
    #     args.teamsvecs = ds
    #     argss.append(copy.copy(args))
    # with mp.Pool(mp.cpu_count()) as p:  p.map(run_entire_ds, argss)

    # for args.teamsvecs in datasets:
    #     args.output = args.teamsvecs
    #     args.model = 'gnn.n2v'
    #     run_entire_ds(args)

    run_entire_ds(args)