# -*- coding: utf-8 -*-

import pickle, logging
from tqdm import tqdm

log = logging.getLogger(__name__)

# teams as documents, members/skills as words
# embtype = 'member' --> doc_list = ['m1 m2 m3','m2 m4'] or
# embtype = 'skill' --> doc_list = ['s1 s2', 's3'] or
# embtype = 'skillmember' --> doc_list = ['s1 s2 m1 m2 m3','s3 m2 m4']
# embtype = 'skilltime' --> doc_list = ['s1 s2 dt1988','s3 dt1990']
# label_list = ['t1','t2','t3']

from pkgmgr import *
from .team2vec import Team2Vec
#Document-based neural net (Dnn) vs. Graph neural net (Gnn) :)

class Dnn(Team2Vec):
    gensim = None
    def __init__(self, dim, output, device, cgf): super().__init__(dim, output, device, cgf)

    def _prep(self, teamsvecs, indexes):
        datafile = self.output + f'/{self.cfg.embtype}docs.pkl'
        try:
            log.info(f'Loading teams as docs {datafile}  ...')
            with open(datafile, 'rb') as infile: self.data = pickle.load(infile)
            return self.data
        except FileNotFoundError:
            log.info(f'File not found! Creating the documents out of teams ...')
            self.data = []
            j = 0
            year = indexes['i2y'][0][1]
            for i in range(teamsvecs['skill'].shape[0]): #or ['member'], as we are enumerating rows that shows teams
                skill_doc = [f's{str(skill_idx)}' for skill_idx in teamsvecs['skill'][i, :].nonzero()[1]]
                member_doc = [f'm{str(member_idx)}' for member_idx in teamsvecs['member'][i, :].nonzero()[1]]

                # level up by a trigger
                if j < len(indexes['i2y']) and indexes['i2y'][j][0] == i:
                    year = indexes['i2y'][j][1]
                    year_idx = j  # zero-based
                    j += 1
                datetime_doc = [f'dt{str(year)}']

                if   self.cfg.embtype == 'skill': td = Dnn.gensim.models.doc2vec.TaggedDocument(skill_doc, [str(i)])
                elif self.cfg.embtype == 'member': td = Dnn.gensim.models.doc2vec.TaggedDocument(member_doc, [str(i)])
                elif self.cfg.embtype == 'skillmember': td = Dnn.gensim.models.doc2vec.TaggedDocument(skill_doc + member_doc, [str(i)])
                elif self.cfg.embtype == 'skilltime': td = Dnn.gensim.models.doc2vec.TaggedDocument(skill_doc + datetime_doc, [str(i)])
                self.data.append(td)
            assert teamsvecs['skill'].shape[0] == len(self.data)
            log.info(f'{len(self.data)} documents with word type of {self.cfg.embtype} have created. Saving ...')
            with open(datafile, 'wb') as f: pickle.dump(self.data, f)
            return self.data

    def train(self, epochs, teamsvecs, indexes):
        # to select/create correct model file in the output directory
        output = self.output + f'/{self.cfg.embtype}.d{self.dim}.w{self.cfg.window}.dm{self.cfg.dm}.e{epochs}.mdl'
        try:
            log.info(f"Loading the model {output} for {(teamsvecs['skill'].shape[0], self.dim)}  embeddings ...")
            Dnn.gensim = install_import('gensim==4.3.3', 'gensim')
            self.model = Dnn.gensim.models.Doc2Vec.load(output)
            assert self.model.docvecs.vectors.shape[0] == teamsvecs['skill'].shape[0] # correct number of embeddings per team
            log.info(f'Model of {self.model.docvecs.vectors.shape} embeddings loaded.')
            return self.model
        except FileNotFoundError:
            log.info(f'File not found! Training the embedding model from scratch ...')
            self._prep(teamsvecs, indexes)
            self.model = Dnn.gensim.models.Doc2Vec(dm=self.cfg.dm, vector_size=self.dim,
                                               window=self.cfg.window, dbow_words=1, min_alpha=0.025, min_count=1, seed=0, workers=self.cfg.nworkers)

            self.model.build_vocab(self.data)
            for e in tqdm(range(epochs)):
                self.model.train(self.data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
                self.model.alpha -= 0.002  # decrease the learning rate
                self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

            log.info(f'Saving model at {output} ...')
            self.model.save(output)
            # self.model.save_word2vec_format(f'{output}.w2v')
            # self.model.docvecs.save_word2vec_format(f'{output}.d2v')
            return self.model
        except Exception as e: raise e

    # NOTE: As the input is a skill subset, the team<->doc vecs is used
    # TODO: If sum/avg of each indivisual skill vecs is desired

    # depending on embtype:
    # words can be skills ['s2', 's5'], members ['m1', 'm5'], skillmember ['s3', 'm4'], skillyear ['dt1996', 's1'],
    def infer_vec(self, words):
        iv = self.model.infer_vector(words)
        return iv, self.model.docvecs.most_similar([iv])


# # unit tests :D
# if cfg.model.d2v.embtype == 'skill': print(t2v.model['s5'])
# if cfg.model.d2v.embtype == 'member': print(t2v.model['m5'])
# if cfg.model.d2v.embtype == 'joint':
#     print(t2v.model['s5'])
#     print(t2v.model['m5'])
# print(t2v.model.docvecs[10])#teamid
# print(t2v.infer_skillsubsetvec(['s1', 's5']))
# print(t2v.skillsubsetvecs().shape)
