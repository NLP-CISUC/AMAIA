import keras as k
import pandas as pd
import os
import cufflinks as cf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import  Embedding
from keras.layers import Dense,SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import LSTM
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.models import Model, load_model
from sklearn.utils import resample
import tensorflow as tf
from keras.optimizers import adam,RMSprop
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
#naive bayes, svm, random forest



#https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb

#1---Empresa na Hora
#2---Marca na hora e marca na hora online
#3---Formas jurídicas
#4---Cartão de Empresa/Cartão de Pessoa Coletiva
#5---Criação da Empresa Online
#6---Certificados de Admissibilidade
#7---Inscrições Online
#8---Certidões Online
#9---Gestão da Empresa Online
#10---RJACSR
#11---Alojamento Local

def escala(X,y):
	from imblearn.over_sampling import RandomOverSampler
	sampler = RandomOverSampler(sampling_strategy='not majority',random_state=0)
	X_train, Y_train = sampler.fit_sample(X, y)
	return X_train,Y_train

def treina(model_name):

	df = pd.read_csv("Input/joined_file_binary_v3.txt",sep='§',header=0)
	df.info()

	max_len = 0
	for value in df.Perguntas:
		if(len(value)>max_len):
			#print(value)
			max_len = len(value)
	max_words = 0
	for value in df.Perguntas:
		word_count = len(value.split(" "))
		if(word_count>max_words):
			#print(word_count)
			max_words = word_count
	#print("---------")
	#print(max_words)
	#print(df.Class.value_counts().sort_values())
	g = df.groupby('Class')
	g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	#print(g)

	#df =  pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	#print("###")
	#print(df.shape)
	#print(df.head(10))
	#df = g



	#X_conveted = pd.get_dummies(df["Perguntas"])
	#Divisão em treino e teste
	X_train, X_test, Y_train, Y_test = train_test_split(df["Perguntas"],df["Class"], test_size = 0.3, random_state = 42)
	#print(X_train.shape)
	#print(Y_train.shape)
	#print(X_test.shape)
	#print(Y_test.shape)



	vect = TfidfVectorizer().fit(X_train)

	with open("Models/vect_bin", 'wb') as fid:
		pickle.dump(vect, fid)	

	X_train_vectorized = vect.transform(X_train)

	X_train_vectorized, Y_train = escala(X_train_vectorized, Y_train)

	clf = svm.SVC(gamma="scale",kernel='linear', degree=16)
	clf.fit(X_train_vectorized,Y_train)

	with open(model_name, 'wb') as fid:
		pickle.dump(clf, fid)	

	preds = clf.predict(vect.transform(X_test))
	score = classification_report(Y_test, preds,target_names = ["OOD","ID"])
	print(score)

	return vect

def corre_para_ficheiro(modelo,ficheiro):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = []
	with open(ficheiro,"r") as g:
		for line in g:
			sentences.append(line.replace("\n","").split("§")[0])
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)


	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		#print(value)
		if(value==1):
			in_domain.append(sentences[index])
		#else:
		#	print(sentences[index])
	return in_domain

def corre_para_frase(modelo,frase):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open(os.path.dirname(os.path.realpath(__file__)) + "/Models/vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = [frase]
	#print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)

	return preds[0]

	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		#print(value)
		if(value==1):
			in_domain.append(frase)
	return in_domain

def corre_para_testes(modelo,ficheiro):
	print("##############")
	nome = ficheiro.replace("out_","")
	nome = nome.replace(".txt","")
	print(nome.upper())
	print("##############")
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = []
	true_results = []
	classe_original = []
	with open(ficheiro,"r") as f:
		for line in f:
			line = line.replace("\n","")
			line = line.split("§")
			sentences.append(line[0])
			classe_original.append(int(line[1]))
			if(int(line[1])==0):
				true_results.append(0)
			else:
				true_results.append(1)
	#print(true_results)
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)

	score = classification_report(true_results, preds,target_names = ["OOD","ID"])
	print(score)


	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		if(value==1):
			frase = str(sentences[index])+"§"+str(classe_original[index])
			in_domain.append(frase)
	return in_domain

def corre(modelo,v):
	labels = ["In the domain","In the domain","Formas jurídicas","Cartão de Empresa/Cartão de Pessoa Coletiva","Criação da Empresa Online","Certificados de Admissibilidade","Inscrições Online","Certidões Online","Gestão da Empresa Online","RJACSR","Alojamento Local","Out of Domain","Out of Domain","Out of Domain","Out of Domain","Out of Domain","Out of Domain"]
	
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	a = 0

	while (a==0):
		print("Entrada.")
		entrada = input()
		entrada = [entrada]
		#entrada = pd.DataFrame([entrada])
		print("Entrada recebida.")
		transformada = v.transform(entrada)
		preds = clfrNB.predict(transformada)
		print(preds)
		if(preds[0]==1):
			print("ID")
		else:
			print("OOD")



if __name__ == '__main__':
	
	treina("Models/svm_binaria_v3.pickle")
	
	print("################ Resultados VIN ##################")
	corre_para_testes("Models/svm_binaria_v3.pickle","Input/Inputs3/VIN_v2.txt")
	print("################ Resultados VG1 ##################")
	corre_para_testes("Models/svm_binaria_v3.pickle","Input/Inputs3/VG1_v2.txt")
	print("################ Resultados VG2 ##################")
	corre_para_testes("Models/svm_binaria_v3.pickle","Input/Inputs3/VG2_v2.txt")
	print("################ Resultados VUC ##################")
	corre_para_testes("Models/svm_binaria_v3.pickle","Input/Inputs3/VUC_v2.txt")
	print("################ Resultados Total ##################")
	corre_para_testes("Models/svm_binaria_v3.pickle","Input/Inputs3/total.txt")
	#corre_para_ficheiro("Models/svm_binaria_v3.pickle","Input/Inputs3/total.txt")

