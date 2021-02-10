from pathlib import Path
import os
import importlib

from agents.ChitchatAgent.ChitchatAgent import ChitchatAgent
from agents.DefaultAgent.DefaultAgent import responseMap, DefaultAgent
from agents.STSAgent.STSAgent import STSAgent
from agents.WhooshAgent.WhooshAgent import WhooshAgent
from agents.Word2vecAgent.Word2vecAgent import Word2vecAgent
from agents.BertAgent.BertAgent import BertAgent
from decisionStrategies.BordaStrategy import BordaStrategy
from decisionStrategies.SimpleMajority import SimpleMajority

"""
General Agent Class - Creates a class that loads all agents
"""

class GeneralAgent(DefaultAgent):

    def __init__(self, corpora, map_resp, agent_names, decision='borda', w2v_model=None, bert_model=None):

        super().__init__(corpora, map_resp)

        self.decision_strategy = BordaStrategy() if decision == 'borda' else SimpleMajority()
        self.map_responses = responseMap(corpora)
        self.agents = startAgents(agent_names, corpora, map_resp=self.map_responses, w2v_model=w2v_model, bert_model=bert_model)
        #for agent in self.agents:
        #    print("Agent: <" + agent.getName() + "> Loaded")


    #Function to get a specific agent by its name
    def getAgent(self, name):
        for agent in self.agents:
            if agent.getName() == name:
                return agent
            else:
                pass
        return None


    #Function to pop a Agent from agents variable from a given name
    def popAgent(self, name):
        for agent in self.agents:
            if agent.getName() == name:
                self.agents.remove(agent)
                return agent


    def matching_questions(self, input, num_answer=5, theta=0.1):
        answers = []
        for agent in self.agents:
            answers.append(agent.matching_questions(input, num_answer, theta))

        best = self.decision_strategy.getNAnswers(answers, num_answer)
        return best


    # Auxiliary function to retrieve a corpus response based on a question
    def find_response(self, interaction):
        return self.map_responses[interaction]
        #response = "[" + self.response_map[interaction]['S'] + "] " + self.response_map[interaction]['R']
        #return response


def startAgents(agent_names, corpora, map_resp=None, w2v_model=None, bert_model=None):
    agent_list = []

    #Looping through agent filenames
    for name in agent_names:
        agent = create_single_agent(name, corpora, map_resp, w2v_model=w2v_model, bert_model=bert_model)
        agent_list.append(agent)

    return agent_list


def create_single_agent(name, corpus, map_resp=None, w2v_model=None, bert_model=None):

    if name == 'Bert':
        return BertAgent(corpus, map_resp, bert_model)
    elif name == 'W2V':
        return Word2vecAgent(corpus, map_resp, w2v_model)
    elif name == 'Whoosh':
        return WhooshAgent(corpus, map_resp)
    elif name == 'ChitChat':
        return ChitchatAgent(corpus)
    elif name == 'STS':
        return STSAgent(corpus, map_resp)

    print("[ERROR] Unknown agent type:", name)
    return None


#Function to get class files from a specific directory. Goes by depth
def get_config_files(dirName):
    directories = os.listdir(dirName)
    configFiles = []
    for d in directories:
        fullpath = os.path.join(dirName, d)
        if os.path.isdir(fullpath):
            configFiles = configFiles + get_config_files(fullpath)
        elif fullpath.endswith('.py'):
            #Pycharm IDE creates __init__.py files, this is for handling that situation.
            if fullpath.endswith('__init__.py'):
                pass
            else:
                configFiles.append(fullpath)
    return configFiles

