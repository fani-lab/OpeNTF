# -*- coding: utf-8 -*-

import gensim, numpy, pylab, random, pickle
import os, getopt, sys, multiprocessing
from tqdm import tqdm
import argparse

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# teams as documents, members/skills as words
# doc_list = ['m1 m2 m3','m2 m4'] or
# doc_list = ['s1 s2', 's3'] or
# doc_list = ['s1 s2 m1 m2 m3','s3 m2 m4']
# label_list = ['t1','t2','t3']

from team2vec import Team2Vec
class Wnn(Team2Vec):
    def __init__(self, teamsvecs, indexes, settings, output):
        super().__init__(teamsvecs, indexes, settings, output)

    def create(self, file):
        j = 0
        year = self.indexes['i2y'][0][1]
        for i, id in enumerate(self.teamsvecs['id']):
            skill_doc = [f's{str(skill_idx)}' for skill_idx in self.teamsvecs['skill'][i].nonzero()[1]]
            member_doc = [f'm{str(member_idx)}' for member_idx in self.teamsvecs['member'][i].nonzero()[1]]

            # Start Hossein: level up by a trigger
            if j < len(self.indexes['i2y']) and self.indexes['i2y'][j][0] == i:
                year = self.indexes['i2y'][j][1]
                year_idx = j  # zero-based
                j += 1
            # End Hossein
            datetime_doc = [f"dt{str(year)}"]

            if   self.settings["embtype"] == 'skill': td = gensim.models.doc2vec.TaggedDocument(skill_doc, [str(i)])
            elif self.settings["embtype"] == 'member': td = gensim.models.doc2vec.TaggedDocument(member_doc, [str(i)])
            elif self.settings["embtype"] == 'joint': td = gensim.models.doc2vec.TaggedDocument(skill_doc + member_doc, [str(i)])
            elif self.settings["embtype"] == 'dt2v': td = gensim.models.doc2vec.TaggedDocument(skill_doc + datetime_doc, [str(i)])
            self.data.append(td)
        print(f'#Documents with word type of {self.settings["embtype"]} have created: {len(self.data)}')
        print(f'Saving the {self.settings["embtype"]} documents ...')
        with open(file, 'wb') as f: pickle.dump(self.data, f)
        return self.data

    def train(self):
        output = self.output + f'emb.d{self.settings["d"]}.w{self.settings["window"]}.dm{self.settings["dm"]}'
        try:
            print(f"Loading the embedding model {output}  ...")
            self.model = gensim.models.Doc2Vec.load(f'{output}.mdl')
            return self.model
        except FileNotFoundError:
            print(f"File not found! Learning {output}.mdl embeddings from scratch ...")

            self.model = gensim.models.Doc2Vec(dm=self.settings["dm"],
                                               # training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                               vector_size=self.settings["d"],
                                               window=self.settings["window"],
                                               dbow_words=self.settings["dbow_words"],
                                               # ({1,0}, optional) – If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
                                               min_alpha=0.025,
                                               min_count=1,
                                               seed=0,
                                               workers=min(16, multiprocessing.cpu_count())) # capping for moderate server usage

            if not self.data: self.init()
            self.model.build_vocab(self.data)
            for e in tqdm(range(self.settings['max_epochs'])):
                self.model.train(self.data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
                self.model.alpha -= 0.002  # decrease the learning rate
                self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

            print(f'Saving model for {output} ...')
            self.model.save(f'{output}.mdl')
            # self.model.save_word2vec_format(f'{output}.w2v')
            # self.model.docvecs.save_word2vec_format(f'{output}.d2v')
            return self.model
        except Exception as e: raise e

    def dv(self): return self.model.docvecs.vectors_docs

    def infer_d2v(self, words):
        iv = self.model.infer_vector(words)
        return iv, self.model.docvecs.most_similar([iv])


def addargs(parser):
    embedding = parser.add_argument_group('Team Doc2Vec Embedding')
    embedding.add_argument('-teamsvecs', type=str, required=True, help='The path to the teamsvecs.pkl and indexes.pkl files; (e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')
    embedding.add_argument('-dm', type=int, default=1, help='The training algorithm; (1: distributed memory (default), 0: CBOW')
    embedding.add_argument('-dbow_words', type=int, default=0, help='Train word-vectors in skip-gram fashion; (0: no (default), 1: yes')
    embedding.add_argument('-d', type=int, default=100, help='Embedding vector dimension; (100 default)')
    embedding.add_argument('-window', type=int, default=1, help='Coocurrence window; (1 default)')
    embedding.add_argument('-max_epochs', type=int, default=10, help='Training epoch; (10 default)')
    embedding.add_argument('-embtype', type=str, default='skill', help="Embedding types; (-embtypes=skill (default); member; joint; )")
    embedding.add_argument('-output', type=str, required=True, help='Output folder; (e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')

def run(teamsvecs_file, indexes_file, settings, output):
    with open(teamsvecs_file, 'rb') as teamsvecs_f, open(indexes_file, 'rb') as indexes_f:
        teamsvecs, indexes = pickle.load(teamsvecs_f), pickle.load(indexes_f)
        t2v = Wnn(teamsvecs, indexes, settings, output)
        t2v.init()
        t2v.train()

        #unit tests :D
        # if embtype == 'skill': print(t2v.model['s5'])
        # if embtype == 'member': print(t2v.model['m5'])
        # if embtype == 'joint':
        #     print(t2v.model['s5'])
        #     print(t2v.model['m5'])
        # print(t2v.model.docvecs[10])#teamid


#python -u wnn.py
# -teamsvecs=../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl
# -embtype=skill
# -d=100
# -dbow_words=1
# -output=../data/preprocessed/dblp/toy.dblp.v12.json/

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team Doc2Vec Embedding')
    addargs(parser)
    args = parser.parse_args()
    settings = {'dm': args.dm,
                'dbow_words': args.dbow_words,
                'd': args.d,
                'window': args.windsow,
                'max_epochs': args.max_epochs,
                'embtype': args.embtype}

    run(f'{args.teamsvecs}teamsvec.pkl', f'{args.teamsvecs}indexes.pkl', settings, args.output)

