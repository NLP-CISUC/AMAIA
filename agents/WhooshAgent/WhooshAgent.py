import os, os.path
from whoosh.index import create_in
import whoosh.index as index
from whoosh.fields import *
from whoosh.query import FuzzyTerm, Term
from whoosh.qparser import QueryParser, OrGroup

from agents.DefaultAgent.DefaultAgent import DefaultAgent

'''
Whoosh Agent logic

Uses Whoosh index for retrieving the most similar awnsers to a query
'''


class WhooshAgent(DefaultAgent):

    def __init__(self, corpus, map_resp, index_path='index_whoosh'):

        super().__init__(corpus, map_resp)
        # self.agentName = self.__class__.__name__
        # self.corpus = corpus

        self.import_modules = []
        self.agents_name = []
        self.index_dir = index_path
        self.indexa_whoosh(self.map_responses)

        print("Agent: <" + self.agentName + "> Loaded")


    # requestAnswer is mandatory, recieves user Input and retuns lists of most similar awnsers
    def matching_questions(self, userInput, num_answer=5, theta=0.1):
        answer = self.query_whoosh(userInput, maxres=num_answer)[0:num_answer]
        ret_answer = []
        for r in answer:
            if(r[2]>=(answer[0][2])*(1-theta)):
                ret_answer.append(r[0])

        return ret_answer

    # Function to query Whoosh index to retrieve the most N similar awnsers
    def query_whoosh(self, q, maxres, fuzzy=False):
        ix = index.open_dir(self.index_dir)
        with ix.searcher() as searcher:
            query = QueryParser("p", ix.schema, group=OrGroup, termclass=(FuzzyTerm if fuzzy else Term)).parse(q)
            results = searcher.search(query, limit=maxres)
            list_results = []
            for r in results:
                list_results.append((r.get("p"), r.get("r"),r.score))
            return list_results


    # Function to create whoosh index based on corpus input
    def indexa_whoosh(self, mapa):
        dir = self.index_dir
        if not os.path.exists(dir):
            os.mkdir(dir)

        # schema = Schema(s=TEXT(stored=True,phrase=False), p=TEXT(stored=True,phrase=False), r=TEXT(stored=True,phrase=False))
        # analyzer=LanguageAnalyzer("pt")
        schema = Schema(s=TEXT(stored=True), p=TEXT(stored=True), r=TEXT(stored=True))
        ix = create_in(dir, schema)
        writer = ix.writer()
        for p in mapa.keys():
            e = mapa[p]
            writer.add_document(s=e['S'], p=p, r=e['R'])
        writer.commit()
        print("Indexação concluída!")

'''
    # Function to create a dict of AIA-BDE corpus: question -> answer
    def mapaPR(self, dataset):
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
        mapa = {}
        with open(fich, "r") as corpus:
            s = None
            p = None
            for line in corpus:
                dp = line.index(":")
                tag = line[:dp]
                content = line[dp + 1:].strip()
                if tag == "S":
                    s = content
                elif tag == "P":
                    p = content
                elif tag == "R":
                    mapa[p] = (content, s)
        return mapa
'''