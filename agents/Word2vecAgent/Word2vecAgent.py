from numpy.linalg import norm
from gensim.models import KeyedVectors

import os
import numpy as np
import string
import time

from agents.DefaultAgent.DefaultAgent import DefaultAgent

'''
Word2Vec Agent logic

Uses W2V word embeddings for sentence transformation and to calculate similarity between them

Class constructor, calls load_w2v_model function to load W2V Model.
Final transformation is performed with mapa_vec function
which returns a new dictionary of corpus Question and its Bert embedding.
'''

class Word2vecAgent(DefaultAgent):

    def __init__(self, corpus, mapa_resp, model_name):

        super().__init__(corpus, mapa_resp)

        self.import_modules = []
        self.agents_name = []

        #self.mapa = self.mapaVars(corpus)

        self.w2v_model = self.load_w2v_model(model_name)
        self.mapa_vec = self.mapa_w2v(list(self.map_responses.keys()), self.w2v_model)

        print("Agent: <" + self.agentName + "> Loaded")


    # requestAnswer is mandatory, receives user Input and retuns list of most similar answers
    def matching_questions(self, userInput, num_answer=5, theta=0.1):
        answer = self.query_w2v(userInput, self.w2v_model, self.mapa_vec, num_answer)

        ret_answer = []
        # Acessing awnsers text, r[1] is answer rank
        for r in answer:
            if(r[1]>=(answer[0][1])*(1-theta)):
                ret_answer.append(r[0])

        return ret_answer


    # Create a dictionary where key is a question and the value its the question words average of W2V embedding.
    def mapa_w2v(self, lista, model):
        print("Criar mapa word2vec...")
        mapa_vec = {}
        for p in lista:
            vec = self.rep_w2v(p, model)
            # print(p[0], vec)
            mapa_vec.setdefault(p, vec)
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


    # Loading w2v model funtion, must train a new model if now available
    def load_w2v_model(self, model_name):
        model_load_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_name)
        start_time = time.time()
        print("Loading word2vec model...", model_name)
        word2vec_model = KeyedVectors.load_word2vec_format(model_load_path)
        print("Model loaded")
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\a')

        return word2vec_model