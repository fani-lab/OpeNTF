# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 21 2019
@author: Hossein Fani (sites.google.com/site/hosseinfani/)
"""

import gensim, numpy, pylab, random, pickle
import os, getopt, sys, multiprocessing
from tqdm import tqdm
import argparse

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# teams as documents, members/skills as words
# doc_list = ['m1 m2 m3','m2 m3','m1 m2 m1 m2'] or ['s1 s3 s4', 's3 s6']
# label_list = ['t1','t2','t3']

class Team2Vec:
    def __init__(self, teamsvecs, embtype, output):
        self.output = output
        self.embtype = embtype
        self.docs = []
        self.settings = ''
        self.teamsvecs = teamsvecs

    def init(self):
        try:
            print(f"Loading the {self.embtype} documents pickle ...")
            with open(f'{self.output}/{self.embtype}.docs.pkl', 'rb') as infile:
                self.docs = pickle.load(infile)
                return self.docs
        except FileNotFoundError:
            print(f"File not found! Generating {self.embtype} documents ...")
            for i, id in enumerate(self.teamsvecs['id']):
                skill_doc = [f's{str(skill_idx)}' for skill_idx in self.teamsvecs['skill'][i].nonzero()[1]]
                member_doc = [f'm{str(member_idx)}' for member_idx in self.teamsvecs['member'][i].nonzero()[1]]
                if self.embtype == 'skill':
                    td = gensim.models.doc2vec.TaggedDocument(skill_doc, [str(int(id[0, 0]))])
                elif self.embtype == 'member':
                    td = gensim.models.doc2vec.TaggedDocument(member_doc, [str(int(id[0, 0]))])
                elif self.embtype == 'joint':
                    td = gensim.models.doc2vec.TaggedDocument(skill_doc + member_doc, [str(int(id[0, 0]))])
                self.docs.append(td)
            print(f'#Documents with word type of {self.embtype} have created: {len(self.docs)}')
            print(f'Saving the {self.embtype} documents ...')
            with open(f'{self.output}/{self.embtype}.docs.pkl', 'wb') as f:
                pickle.dump(self.docs, f)
            return self.docs

    def train(self, dimension=300, window=1, dm=1, dbow_words=0, epochs=10):
        self.settings = f'{self.embtype}.emb.d{dimension}.w{window}.dm{dm}'
        try:
            print(f"Loading the {self.embtype} embedding pickle ...")
            self.model = gensim.models.Doc2Vec.load(f'{self.output}/{self.settings}.mdl')
            return self.model
        except FileNotFoundError:
            print(f"File not found! Learning {self.settings} embeddings from scratch ...")

            self.model = gensim.models.Doc2Vec(dm=dm,
                                               # training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                               vector_size=dimension,
                                               window=window,
                                               dbow_words=dbow_words,
                                               # ({1,0}, optional) – If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
                                               min_alpha=0.025,
                                               min_count=0,
                                               seed=0,
                                               workers=multiprocessing.cpu_count())

            if not self.docs: self.init()
            self.model.build_vocab(self.docs)
            for e in tqdm(range(epochs)):
                self.model.train(self.docs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
                self.model.alpha -= 0.002  # decrease the learning rate
                self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

            print(f'Saving model for {self.settings} under directory {self.output} ...')
            self.model.save(f'{self.output}/{self.settings}.mdl')
            # self.model.save_word2vec_format(f'{output}/{self.settings}.w2v')
            # self.model.docvecs.save_word2vec_format(f'{output}/{self.settings}.d2v')
            return self.model
        except Exception as e:
            raise e

    def dv(self):
        return self.model.docvecs.vectors_docs

    def infer_d2v(self, words):
        iv = self.model.infer_vector(words)
        return iv, self.model.docvecs.most_similar([iv])


def addargs(parser):
    embedding = parser.add_argument_group('embedding')
    embedding.add_argument('-teamsvecs', type=str, required=True, help='The path to the teamsvecs.pkl file; (e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl')
    embedding.add_argument('-dm', type=int, default=1, help='The training algorithm; (1: distributed memory (default), 0: CBOW')
    embedding.add_argument('-dbow_words', type=int, default=0, help='Train word-vectors in skip-gram fashion; (0: no (default), 1: yes')
    embedding.add_argument('-dim', type=int, default=100, help='Embedding vector dimension; (100 default)')
    embedding.add_argument('-window', type=int, default=1, help='Coocurrence window; (1 default)')
    embedding.add_argument('-epochs', type=int, default=10, help='Training epoch; (10 default)')
    embedding.add_argument('-embtypes', type=str, default='skill', help="Embedding type; (-embtypes=skill (default); member; joint; skill,member; skill, joint; ...)")
    embedding.add_argument('-output', type=str, required=True, help='Output folder; (e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')

def run(teamsvecs_file, dm, dbow_words, dim, window, embtypes, epochs, output):
    with open(teamsvecs_file, 'rb') as infile:
        teamsvecs = pickle.load(infile)
        for embtype in embtypes:
            t2v = Team2Vec(teamsvecs, embtype, output)
            #2v.init()
            t2v.train(dim, window, dm, dbow_words, epochs)

            #unit tests :D
            # if embtype == 'skill': print(t2v.model['s5'])
            # if embtype == 'member': print(t2v.model['m5'])
            # if embtype == 'joint':
            #     print(t2v.model['s5'])
            #     print(t2v.model['m5'])
            # print(t2v.model.docvecs[10])#teamid

#python -u team2vec.py -teamsvecs=../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl -embtypes=skill,member,joint -dim=100 -dbow_words=1 -output=../data/preprocessed/dblp/toy.dblp.v12.json/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team Embedding')
    addargs(parser)
    args = parser.parse_args()
    run(args.teamsvecs, args.dm, args.dbow_words, args.dim, args.window, args.embtypes.split(','), args.epochs, args.output)

