import pickle, logging, os, numpy as np

log = logging.getLogger(__name__)

# teams as documents, members/skills as words
# embtype = 'member' --> doc_list = ['m1 m2 m3','m2 m4'] or
# embtype = 'skill' --> doc_list = ['s1 s2', 's3'] or
# embtype = 'skillmember' --> doc_list = ['s1 s2 m1 m2 m3','s3 m2 m4']
# embtype = 'skilltime' --> doc_list = ['s1 s2 dt1988','s3 dt1990']
# label_list = ['t1','t2','t3']

import pkgmgr as opentf
from .t2v import T2v

class D2v(T2v):
    gensim = None

    def _prep(self, teamsvecs, splits=None, time_indexes=None):
        datafile = self.output + f'/../{self.cfg.embtype}.docs.pkl'
        if self.cfg.embtype == 'skilltime': assert time_indexes, f'{opentf.textcolor["red"]}Temporal skill embedding needs time indexes!{opentf.textcolor["reset"]}'

        try:
            log.info(f'Loading teams as docs {datafile}  ...')
            with open(datafile, 'rb') as infile: self.data = pickle.load(infile)
            return self
        except FileNotFoundError:
            log.info(f'File not found! Creating the documents out of teams ...')
            self.data = []
            j = 0
            year = time_indexes['i2y'][0][1] if self.cfg.embtype == 'skilltime' else None
            for i in range(teamsvecs['skill'].shape[0]): #or ['member'], as we are enumerating rows that shows teams
                skill_doc = [f's{str(skill_idx)}' for skill_idx in teamsvecs['skill'][i, :].nonzero()[1]]
                member_doc = [f'm{str(member_idx)}' for member_idx in teamsvecs['member'][i, :].nonzero()[1]]

                # level up by a trigger
                if year and j < len(time_indexes['i2y']) and time_indexes['i2y'][j][0] == i:
                    year = time_indexes['i2y'][j][1]
                    year_idx = j  # zero-based
                    j += 1
                datetime_doc = [f'dt{str(year)}']

                if   self.cfg.embtype == 'skill': td = self.gensim.models.doc2vec.TaggedDocument(skill_doc, [str(i)])
                elif self.cfg.embtype == 'member': td = self.gensim.models.doc2vec.TaggedDocument(member_doc, [str(i)])
                elif self.cfg.embtype == 'skillmember': td = self.gensim.models.doc2vec.TaggedDocument(skill_doc + member_doc, [str(i)])
                elif self.cfg.embtype == 'skilltime': td = self.gensim.models.doc2vec.TaggedDocument(skill_doc + datetime_doc, [str(i)])
                self.data.append(td)
            assert teamsvecs['skill'].shape[0] == len(self.data), f'{opentf.textcolor["red"]}Incorrect number of docs for teams! {teamsvecs["skill"].shape[0]} != {len(self.data)}{opentf.textcolor["reset"]}'
            log.info(f'{len(self.data)} documents with word type of {self.cfg.embtype} have created. Saving ...')
            with open(datafile, 'wb') as f: pickle.dump(self.data, f)
            return self

    def learn(self, teamsvecs, splits=None, time_indexes=None):
        # to select/create correct model file in the output directory
        modelstr = f'{self.name}.d{self.cfg.d}.e{self.cfg.e}.w{self.cfg.w}.dm{self.cfg.dm}.{self.cfg.embtype}'
        modelpath = f'{self.output}/{modelstr}'
        modelfile = f'{self.output}/{modelstr}/{modelstr}.pt'
        try:
            log.info(f'Loading the model {modelfile} for {(teamsvecs["skill"].shape[0], self.cfg.d)} embeddings ...')
            self.__class__.gensim = opentf.install_import('gensim')
            self.model = self.gensim.models.Doc2Vec.load(modelfile)
            assert self.model.docvecs.vectors.shape[0] == teamsvecs['skill'].shape[0], f'{opentf.textcolor["red"]}Incorrect number of embeddings per team! {self.model.docvecs.vectors.shape[0]} != {teamsvecs["skill"].shape[0]}{opentf.textcolor["reset"]}'
            self.output = modelpath
            return self
        except FileNotFoundError:
            log.info(f'{opentf.textcolor["yellow"]}File not found! Training the embedding model from scratch ...{opentf.textcolor["reset"]}')
            self._prep(teamsvecs, splits, time_indexes)
            self.output = modelpath
            if not os.path.isdir(self.output): os.makedirs(self.output)
            self.model = self.gensim.models.Doc2Vec(min_count=1, dbow_words=1, # keep it always one as it may be needed for gnn-based method for 'pre' config, i.e., initial node features
                                                    dm=self.cfg.dm, vector_size=self.cfg.d, window=self.cfg.w, min_alpha=self.cfg.lr,
                                                    workers=self.device.split(':')[1] if 'cpu:' in self.device else os.cpu_count() - 1, ** {'seed': self.seed} if self.seed is not None else {})

            self.model.build_vocab(self.data)
            if self.cfg.spe:
                import random
                random.shuffle(self.data)
                for epoch in range(self.cfg.e):
                    self.model.train(self.data, total_examples=self.model.corpus_count, epochs=1)
                    delta = (self.model.alpha - self.model.min_alpha) / (self.cfg.e - 1)
                    self.model.alpha = max(self.model.alpha - delta, self.model.min_alpha)
                    if epoch == 0 or ((epoch + 1) % self.cfg.spe) == 0:
                        log.info(f'Saving model at {modelfile}.{opentf.textcolor["blue"]}e{epoch} at lr {self.model.alpha}{opentf.textcolor["reset"]}...')
                        self.model.save(modelfile.replace('.pt', f'.e{epoch}.pt'))
            else: self.model.train(self.data, total_examples=self.model.corpus_count, epochs=self.cfg.e)

            log.info(f'Saving model at {modelfile} ...')
            self.model.save(modelfile)
            # self.model.save_word2vec_format(f'{output}.w2v')
            # self.model.docvecs.save_word2vec_format(f'{output}.d2v')
            return self
        except Exception as e: raise e

    # NOTE: As the input is a skill subset, the team<->doc vecs is used
    # TODO: If sum/avg of each indivisual skill vecs is desired

    # depending on embtype:
    # words can be skills ['s2', 's5'], members ['m1', 'm5'], skillmember ['s3', 'm4'], skillyear ['dt1996', 's1'],
    def infer_vec(self, words):
        iv = self.model.infer_vector(words)
        return iv, self.model.docvecs.most_similar([iv])

    @staticmethod
    def natsortvecs(d2v_model_wv):
        import numpy as np
        natsorted = opentf.install_import('natsort', from_module='natsorted')
        sorted_words = natsorted(d2v_model_wv.index_to_key)  # ['m3', 's10', 's2', 's1'] --> ['m3', 's1', 's2', 's10']
        sorted_indices = np.array([d2v_model_wv.key_to_index[word] for word in sorted_words])
        return d2v_model_wv.vectors[sorted_indices]

    def get_dense_vecs(self, teamsvecs, vectype='skill'):
        # note that a doc includes a subset of skills or members. So, the doc vec represents the subset.
        # agg='': we can have vec for each individual words (skill or member) but then we have to sum/mean then for a subset. that's w2v, not d2v-based skills or members
        assert self.cfg.embtype == vectype, f'{opentf.textcolor["red"]}Incorrect d2v model ({self.cfg.embtype}) for the requested vector type {vectype}{opentf.textcolor["reset"]}'
        indices = [self.model.docvecs.key_to_index[str(i)] for i in range(len(self.model.docvecs))]
        assert np.allclose(self.model.docvecs.vectors, self.model.docvecs.vectors[indices]), f'{opentf.textcolor["red"]}Incorrect embedding for a team due to misorderings of embeddings!{opentf.textcolor["reset"]}'
        return self.model.docvecs.vectors

# # unit tests :D
# if cfg.model.d2v.embtype == 'skill': print(t2v.model['s5'])
# if cfg.model.d2v.embtype == 'member': print(t2v.model['m5'])
# if cfg.model.d2v.embtype == 'joint':
#     print(t2v.model['s5'])
#     print(t2v.model['m5'])
# print(t2v.model.docvecs[10])#teamid
# print(t2v.infer_skillsubsetvec(['s1', 's5']))
# print(t2v.skillsubsetvecs().shape)
