from numpy.linalg import norm
from gensim.models import KeyedVectors
from ASAPPpy.sts_model import STSModel
from ASAPPpy.scripts.load_embeddings import load_embeddings_models

import os
import numpy as np
import string
import time

from agents.DefaultAgent.DefaultAgent import DefaultAgent

'''
STS Agent logic

Uses an STS model based on ASAPPpy for sentence transformation and to calculate similarity between them

Class constructor, calls load_model function to load STS Model.  
Final transformation is performed with mapa_vec function.
'''

class STSAgent(DefaultAgent):
    def __init__(self, corpus, map_resp=None):

        super().__init__(corpus, map_resp)

        self.import_modules = []
        self.agents_name = []
        self.model = STSModel()
        self.model.load_model('model_R_pos_adv-dependency_parsing-word2vec-ptlkb-numberbatch')
        self.word2vec_model, self.fasttext_model, self.ptlkb64_model, self.glove300_model, self.numberbatch_model = load_embeddings_models()
        #self.mapa = self.mapaVars(corpus)
        #self.mapa_responses = self.mapaResponses(corpus)
        #self.mapa_vec = self.mapa_w2v(self.mapa.get("P"), self.w2v_model)
        print("Agent: <" + self.agentName + "> Loaded")

    # requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def matching_questions(self, userInput, num_answer=5, theta=0.1):
        questions = list(self.map_responses.keys())

        corpus_pairs = []

        for question in questions:
            corpus_pairs.extend([question, userInput])

        element_features = self.model.extract_multiple_features(corpus_pairs, 0, word2vec_mdl=self.word2vec_model, fasttext_mdl=self.fasttext_model, ptlkb_mdl=self.ptlkb64_model, glove_mdl=self.glove300_model, numberbatch_mdl=self.numberbatch_model)

        predicted_similarity = self.model.predict_similarity(element_features)
        predicted_similarity = predicted_similarity.tolist()

        highest_match = max(predicted_similarity)

        #answer = self.query_w2v(userInput, self.w2v_model, self.mapa_vec,num_answer)

        ret_answer = []
        # Acessing awnsers text, r[1] is answer rank
        for pos, pred in enumerate(predicted_similarity):
            if pred >= highest_match*(1-theta):
                ret_answer.append(questions[pos])

        return ret_answer


    # Create a dictionary where key is a question and the value its the question words average of W2V embedding.
    def mapa_w2v(self, lista, model):
        print("Criar mapa word2vec...")
        mapa_vec = {}
        for p in lista:
            vec = self.rep_w2v(p[0], model)
            # print(p[0], vec)
            mapa_vec.setdefault(p[0], vec)
        return mapa_vec

    # Function to represent a sentence to a Word embedding vector by calculating the embedding of each word and perform a average of all word vectors
    def rep_w2v(self, frase, model):
        tokens = self.tokenize(frase)
        # print(tokens)
        vec = np.zeros(len(model['palavra']))
        n = 0
        for t in tokens:
            if t in model.vocab:
                n += 1
                vec = [x + y for x, y in zip(vec, model[t])]
        if n > 0:
            vec = [i / n for i in vec]
        return vec

    #Auxiliary function to tokenize a sentence
    def tokenize(self,frase):
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = frase.translate(table)
        # split into words by white space
        words = stripped.lower().split()
        return words

    # Function to enconde a query to bert embedding and find the most N similar sentences using cosine function
    def query_w2v(self, query, model, mapa, top):
        vec = self.rep_w2v(query, model)
        res = []
        for pp in mapa.keys():
            res.append((pp, self.cosine(vec, mapa.get(pp))))
        return sorted(res, key=lambda tup: tup[1], reverse=True)[:top]


'''
    # Function to create a dict of AIA-BDE corpus
    def mapaVars(self, dataset):
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
        mapa = {}
        with open(fich, "r") as corpus:
            p = None
            for line in corpus:
                dp = line.index(":")
                tag = line[:dp]
                content = line[dp + 1:].strip()
                # print(tag[0], tag, content)
                if tag == "P":
                    p = content
                    if tag not in mapa.keys():
                        mapa.setdefault(tag, [])
                    mapa.get(tag).append((content, p))
                elif tag[0] == 'V':
                    if tag not in mapa.keys():
                        mapa.setdefault(tag, [])
                    # print(p, content)
                    mapa.get(tag).append((content, p))
        return mapa
'''