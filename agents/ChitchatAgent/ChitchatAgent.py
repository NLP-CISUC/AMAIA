from agents.WhooshAgent.WhooshAgent import WhooshAgent

'''
Chitchat Agent logic

Uses IR for matching user interactions with stored interactions.
'''

class ChitchatAgent(WhooshAgent):
    def __init__(self, corpus):

        super().__init__(corpus, map_resp=None, index_path='index_chitchat')
        self.agentName = self.__class__.__name__


    def matching_questions(self, userInput, num_answer=1, theta=0):
        answer = self.query_whoosh(userInput, maxres=num_answer)

        if answer:
            return answer[0][1] if answer else None


'''
    def getName(self):
        return self.agentName


    def requestAnswer(self, userInput,num_answer):
        answer = self.query_bert(userInput,self.bert_model,self.mapa_vec,num_answer)

        ret_answer = []
        get_awnser = self.mapa.get("R")

        multiple = False

        #In case we want to return more than one awnser, pass flag to True and adapt login
        if (multiple):
            for r in answer:
                #find the response to the most similar question
                for val in get_awnser:
                    if val[1] == r[0]:
                        ret_answer.append(val[0])

        else:
            # find the response to the most similar question
            for val in get_awnser:
                if val[1] == answer[0][0]:
                    return val[0]

        return ret_answer


    def mapa_bert(self, lista, bert):
        print("Criar mapa BERT...")
        mapa_vec = {}
        ps = [p[0] for p in lista]
        vec = bert.encode(ps)
        for i in range(len(lista)):
            mapa_vec.setdefault(lista[i][0], vec[i])
        return mapa_vec


    # Function to create a dict of chitchat corpus
    def mapaVars(self, dataset):
        fich = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chitchat.txt')
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
                elif tag == 'R':
                    if tag not in mapa.keys():
                        mapa.setdefault(tag, [])
                    # print(p, content)
                    mapa.get(tag).append((content, p))
        return mapa

    # Function to encode a query to bert embedding and find the most N similar sentences using cosine function
    def query_bert(self, query, bert, mapa, top):
        vec = bert.encode([query])[0]
        # vec = bert.encode([query])[0][0] #pooling_strategy=NONE, word embedding for `[CLS]`
        # print("vec=", vec)
        # print("mapa=", mapa)
        res = []
        for pp in mapa.keys():
            res.append((pp, self.cosine(vec, mapa.get(pp))))
        return sorted(res, key=lambda tup: tup[1], reverse=True)[:top]

    # Function to calculate cosine similarity between two vectors (v1 and v2)
    def cosine(self, v1, v2):
        if all(v == 0 for v in v1) or all(v == 0 for v in v2):
            return 0.0
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def load_bert_model(self):
        start_time = time.time()
        print("Started loading the BERT model")
        word_embedding_model = models.Transformer('neuralmind/bert-base-portuguese-cased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

        bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Model loaded")
        print("--- %s seconds ---" % (time.time() - start_time))
        print('\a')

        return bert_model
'''
