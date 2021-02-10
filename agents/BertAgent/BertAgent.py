import os
import numpy as np
import time
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sentence_transformers import models

from agents.DefaultAgent.DefaultAgent import DefaultAgent

'''
Bert Agent logic

Uses Portuguese Bert word embeddings for sentence transformation and to calculate similarity between them

Class constructor, calls load_bert_model function to load Bert Model, processes the corpus, 
Final transformation is performed with mapa_bert function which returns a new dictionary of corpus Question and its Bert embedding.
'''

class BertAgent(DefaultAgent):

    def __init__(self, corpus, map_resp, model_name):

        super().__init__(corpus, map_resp)

        #self.agentName = self.__class__.__name__
        self.bert_model = self.load_bert_model(model_name)
        #self.mapa = self.mapaVars(corpus)
        #self.mapa_vec = self.mapa_bert(self.mapa.get("P"), self.bert_model)
        self.mapa_vec = self.mapa_bert(list(self.map_responses.keys()), self.bert_model)

        print("Agent: <" + self.agentName + "> Loaded")


    #requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def matching_questions(self, userInput, num_answer=5, theta=0.1):
        answer = self.query_bert(userInput,self.bert_model,self.mapa_vec,num_answer)

        ret_answer = []
        #Acessing awnsers text, r[1] is answer rank
        for r in answer:
            if(r[1]>=(answer[0][1])*(1-theta)):
                ret_answer.append(r[0])

        return ret_answer


    #Loading bert model funtion, will download to local memory if executing for the first time
    def load_bert_model(self, model_name):
        start_time = time.time()
        print("Loading BERT model...", model_name)
        word_embedding_model = models.Transformer(model_name)
        # word_embedding_model = models.Transformer('neuralmind/bert-base-portuguese-cased')
        # word_embedding_model = models.Transformer('bert-base-multilingual-cased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Model loaded")
        print("--- %s seconds ---" % (time.time() - start_time))
        #print('\a')

        return bert_model


    #Create a dictionary where key is a question and the value its the question bert embedding.
    def mapa_bert(self, lista, bert):
        print("Criar mapa BERT...")
        mapa_vec = {}
        vec = bert.encode(lista)
        for i in range(len(lista)):
            # mapa_vec.setdefault(lista[i][0], vec[i][0]) #pooling_strategy=NONE, word embedding for `[CLS]`
            mapa_vec[lista[i]] = vec[i]
        return mapa_vec


    #Function to enconde a query to bert embedding and find the most N similar sentences using cosine function
    def query_bert(self, query, bert, mapa, top):
        vec = bert.encode([query])[0]
        res = []
        for pp in mapa.keys():
            res.append((pp, self.cosine(vec, mapa.get(pp))))
        return sorted(res, key=lambda tup: tup[1], reverse=True)[:top]