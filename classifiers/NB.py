import keras as k
import pandas as pd
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
from sklearn.feature_extraction.text import TfidfVectorizer
import os

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

	df = pd.read_csv("Input/joined_file_binary_v5.txt",sep='§',header=0)
	#df = pd.read_csv("divididossop.txt",sep='§',header=0)
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
			print(word_count)
			max_words = word_count
	#print("---------")
	#print(max_words)
	#print(df.Class.value_counts().sort_values())
	g = df.groupby('Class')
	g= pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	#print(g)

	#df =  pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	#print("###")
	#print(df.shape)
	#print(df.head(10))
	#df = g



	#X_conveted = pd.get_dummies(df["Perguntas"])
	#Divisão em treino e teste
	X_train, X_test, Y_train, Y_test = train_test_split(df["Perguntas"],df["Class"], test_size = 0.1, random_state = 42)
	#print(X_train.shape)
	#print(Y_train.shape)
	#print(X_test.shape)
	#print(Y_test.shape)



	vect = TfidfVectorizer(max_features=200,max_df = 0.5).fit(X_train)
	X_train_vectorized = vect.transform(X_train)

	#X_train_vectorized, Y_train = escala(X_train_vectorized, Y_train)
	#X_train_vectorized = bert_model(X_train)
	#X_train_vectorized = pd.DataFrame.from_records(X_train_vectorized)
	X_train_vectorized, Y_train = escala(X_train_vectorized, Y_train)

	#print(vect)
	clfrNB = MultinomialNB(alpha = 0.1)
	clfrNB.fit(X_train_vectorized, Y_train)

	with open(model_name, 'wb') as fid:
		pickle.dump(clfrNB, fid)	
	with open("Models/vect_nb", 'wb') as fid:
		pickle.dump(vect, fid)	

	#X_test_vectorized =  pd.DataFrame.from_records(bert_model(X_test))
	X_test_vectorized = vect.transform(X_test)
	preds = clfrNB.predict(X_test_vectorized)

	#preds = clfrNB.predict(vect.transform(X_test))
	#print("######")
	#print(clfrNB.get_params())
	#print("######")
	score = classification_report(Y_test, preds)
	print(score)
	
	return vect


def corre_para_testes_restantes(modelo,ficheiro,vect):
	nome = ficheiro.replace("out_","")
	nome = nome.replace(".txt","")
	print(nome.upper())
	with open(modelo, 'rb') as fid:
		svm = pickle.load(fid)

	sentences = []
	true_results = []
	with open(ficheiro,"r") as f:
		for line in f:
			line = line.replace("\n","")
			line = line.split("§")
			sentences.append(line[0])
			true_results.append(int(line[1]))

	#X_test_vectorized =  pd.DataFrame.from_records(bert_model(sentences))
	X_test_vectorized = vect.transform(sentences)
	preds = svm.predict(X_test_vectorized)
	preds_probs = svm.predict_proba(X_test_vectorized)

	score = classification_report(true_results, preds)
	print(score)

def corre_para_frase(modelo,frase):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open(os.path.dirname(os.path.realpath(__file__)) + "/Models/vect_nb", 'rb') as fid:
		v = pickle.load(fid)
	sentences = [frase]
	#print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)

	return preds[0]


if __name__ == '__main__':
	#_lr_0.03
	#modelo = "lstm_com_balanceamento_varias_camadas_500_lr_0.03.h5"
	#modelo = "lstm_com_balanceamento_varias_camadas_200_lr_0.03.h5"
	#vect = treina("GaussianNBDefault.pickle")
	#corre("GaussianNB.pickle",vect)
	#vect = treina("NB_so_P.pickle")
	#corre("NB_so_P.pickle",vect,"out_vuc.txt")
	#corre("NB_so_P.pickle",vect,"out_vg1.txt")
	#corre("NB_so_P.pickle",vect,"out_vg2.txt")
	'''
	vect = treina("Models/NB.pickle")	
	
	corre_para_testes_restantes("Models/NB.pickle","Input/Inputs3/VIN_v4.txt",vect)
	corre_para_testes_restantes("Models/NB.pickle","Input/Inputs3/VG1_v4.txt",vect)
	corre_para_testes_restantes("Models/NB.pickle","Input/Inputs3/VG2_v4.txt",vect)
	corre_para_testes_restantes("Models/NB.pickle","Input/Inputs3/VUC_v4.txt",vect)
	corre_para_testes_restantes("Models/NB.pickle","Input/Inputs3/total_v5.txt",vect)
	'''
	print(corre_para_frase("Models/NB.pickle","Frase para teste"))




