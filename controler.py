import sys
import os
import psutil
from classifiers import svm_binaria, random_forest, NB
import GeneralAgent
from agents.BertAgent import BertAgent
from agents.ChitchatAgent.ChitchatAgent import ChitchatAgent
from agents.Word2vecAgent import Word2vecAgent
from decisionStrategies import BordaStrategy
import warnings
import ast
import logging
'''
Controler script for connecting all the agents
'''

'''
def conversation(configs):
    conversation(agent_names=configs['agents'], corpus=configs['corpus'], ood_corpus=configs['out_of_domain_corpus'],
                 c_name=configs['out_of_domain_classifier'], decision=configs['decision'],
                 answers_per_agent=configs['number_of_answers_per_agent'], theta_val=configs['theta'],
                 w2v_model=configs['w2v_embeddings'],  bert_model=configs['bert_embeddings'])
'''

def start_controler(agent_names=None, corpus=None, ood_corpus=None, c_name=None, answers_per_agent=5, decision='borda',
                 theta_val=0.1, w2v_model=None, bert_model=None):

    global agent, chitchat, number_of_answers_per_agent, theta, f_classify, path_classify

    #out_of_domain_corpus = ood_corpus
    #classifier_name = c_name
    number_of_answers_per_agent = answers_per_agent
    theta = theta_val

    #Use only choosen agents
    print("Carregando modelos, pode demorar alguns minutos.")

    map_resp = GeneralAgent.responseMap(corpus)
    agent = GeneralAgent.GeneralAgent(corpus, map_resp, agent_names, decision, w2v_model, bert_model)

    if ood_corpus:
        chitchat = ChitchatAgent(ood_corpus)
        f_classify, path_classify = init_classificador(c_name)

    print("Modelos carregados")
    print("Bem vindo à Amaia...")


#Loop for command-line interaction with the agents
def conversation():

    #Loop for interaction
    while True:
        print('> ', end='')
        input_text = sys.stdin.readline()

        #Logoff
        if len(input_text.strip()) == 0 or input_text.lower().strip() == 'adeus':
            print("Adeus, até uma próxima!")
            break
        else:
            if chitchat:
                #HUGO: Está a usar sempre o mesmo classificador SVM?
                domain = f_classify(os.path.dirname(os.path.realpath(__file__)) + path_classify, input_text)

            else: #se não houver classificador, considera-se sempre do domínio
                domain = 1

            if domain == 1:
                #print("Frase dentro do dominio, ativando agentes")

                matches = agent.matching_questions(input_text, number_of_answers_per_agent, theta)
                for m in matches:
                    service_resp = agent.find_response(m)
                    service = ('[' + service_resp['S'] + '] ') if 'S' in service_resp else ''
                    resp = service_resp['R']
                    print("Se a sua dúvida é \"" + service + str(m) + "\" \n\t* a resposta será: \"" + str(resp) + "\"")

            else:
                #print("Frase fora de dominio, ativando chitchat")
                #Using chichat agent to get an awnser
                response = chitchat.matching_questions(input_text, number_of_answers_per_agent)
                print(response if response else 'Não percebi, pode reformular?')


def init_classificador(classifier_name):
    if not classifier_name or classifier_name == "SVM":
        f_classificar = svm_binaria.corre_para_frase
        path_classificador = '/classifiers/Models/svm_binaria_v3.pickle'
    elif classifier_name == "RF":
        f_classificar = random_forest.corre_para_frase
        path_classificador = '/classifiers/Models/rf_v4.pickle'
    elif classifier_name == "NB":
        f_classificar = NB.corre_para_frase
        path_classificador = '/classifiers/Models/NB.pickle'
    else:
        print("[ERROR] classifiers desconhecido " + classifier_name + ". Valores possíveis: NB, RF, SVM.")
        return None, None

    return f_classificar, path_classificador


#For Flask
def web_chat_interface(input_text):
    if input_text.lower().strip() == 'adeus':
        return("Adeus, até uma próxima!")
    else:
        if f_classify:
            domain = f_classify(os.path.dirname(os.path.realpath(__file__)) + path_classify, input_text)

        else: # se não houver classificador, considera-se sempre do domínio
            domain = 1

        if (domain == 1):
            #print("Frase dentro do dominio, ativando agentes")

            matches = agent.matching_questions(input_text, number_of_answers_per_agent, theta)
            response_full = ''
            for m in matches:
                service_resp = agent.find_response(m)
                service = ('[' + service_resp['S'] + '] ') if 'S' in service_resp else ''
                resp = service_resp['R']
                response_full += 'Se a sua dúvida é <strong>"' + service + str(m) + '"</strong> <br>* a <strong>resposta</strong> será: "' + str(resp) + '"<br>'
                return response_full
        else:
            response = chitchat.matching_questions(input_text)
            return response if response else 'Não percebi, pode reformular?'


def load_config(config_file):
    lines = {}
    with open(config_file,"r") as f:
        contents = f.read()
        lines = ast.literal_eval(contents)
        f.close()
    print(lines)
    return lines


if __name__=="__main__":
    '''
    agents = ["Whoosh"] #Agentes activos - Bert,Whoosh e/ou W2V
    corpus = None #Array dos Corpus para os respectivos agentes
    out_of_domain_corpus = File name #Activar o chat para interações fora de domínio, utilizando este corpus
    out_of_domain_classifier = None #Escolher o modo de classificação para fora do dominio - SVM, NB ou RF
    conversation(agents,corpus,out_of_domain_chat,out_of_domain_classifier)    
    '''

    logging.basicConfig(level=logging.ERROR)
    configs = load_config("config.txt")

    start_controler(configs["agents"], configs["corpus"], configs["out_of_domain_corpus"],
                 configs["out_of_domain_classifier"], configs["number_of_answers_per_agent"], configs["decision"], configs["theta"],
                 configs["w2v_embeddings"], configs["bert_embeddings"])
    conversation()