import os
import numpy as np
from numpy.linalg import norm

'''
Agent with a corpus, but no specified strategy: should not be instantiated
'''

class DefaultAgent:
    def __init__(self, corpus, mapa_resp):
        self.agentName = self.__class__.__name__
        self.corpus = corpus
        self.map_responses = mapa_resp if mapa_resp else responseMap(corpus)


    def getName(self):
        return self.agentName


    #Function to calculate cosine similarity between two vectors (v1 and v2)
    def cosine(self, v1, v2):
        if all(v == 0 for v in v1) or all(v == 0 for v in v2):
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))


    #Not implemented
    def matching_questions(self, userInput, num_answer=5, theta=0.1):
        return None

'''
    Returns a dictionary where interactions map to responses: Interaction -> dict('S', 'R')
'''
def responseMap(corpora):

    map = {}
    for c in corpora:
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), c)
        #print(c, fich)

        with open(fich, "r") as corpus:
            s = None
            p = None
            for line in corpus:
                dp = line.index(":")
                tag = line[:dp]
                content = line[dp + 1:].strip()
                if tag == 'S':
                    s = content
                elif tag == 'P':
                    p = content
                    map[p] = {}
                    map[p]['S'] = s
                elif tag == 'R':
                    map[p]['R'] = content

    #print(len(map.keys()))
    return map