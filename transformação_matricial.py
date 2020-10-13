# encoding: utf-8
# -*- coding: cp1252 -*-.
#!/usr/bin/python
import os
import re
import nltk
import string
import pandas as pd
import numpy as np
import seaborn as sns
import preprocessor as p
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from time import time, sleep
from textblob import TextBlob
from nltk.corpus import stopwords 
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score,recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score, roc_curve, auc
from sklearn.preprocessing import normalize

def lsa_cluster_oneclass ():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

	print('LSA...')
	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	tfidf_lsa=normalize(tfidf_lsa.T)
	tfidf_lsa=tfidf_lsa.T
	
	print("Clustering... KMeans")
	
	true_k = 56
	kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=10000, n_init=1)
	kmeans.fit(tfidf_lsa)
	
	transf_matrix=kmeans.cluster_centers_.T
	short_matrix=np.matmul(tfidf_lsa,transf_matrix)
	
	print("OneClassSVM...")
	df_x = pd.DataFrame(short_matrix[0:,0:])

	print("Treinamento")

	X_train, X_test, y_train, y_test = train_test_split(df_x, df['probability'][0:], test_size=0.1)
	
	train_normal=X_train.loc[df["probability"] == 1]
	train_outliers = X_train.loc[df["probability"] == -1]
	outlier_prop = len(train_outliers)/len(df_x)
	
	gama=0.1
	ker='linear'
	print ('gama:'+ str(gama))
	print ('kernel:'+ str(ker))
	
	clf = OneClassSVM(nu=outlier_prop, kernel=ker,gamma=gama, degree=5)
	clf.fit(train_normal)
	y_pred = clf.predict(X_test)
	
	#for index,row in X_test.iterrows():true.append(df['probability'][index])
	probabilidades = max(clf.decision_function(X_test)) - clf.decision_function(X_test)
	for i in range(len(probabilidades)):probabilidades[i]=1-probabilidades[i]
	print(probabilidades)
	fpr, tpr, thresholds = roc_curve(y_test, probabilidades)
	roc_auc = auc(fpr, tpr)
	g=plt.figure(3)
	plt.rcParams['axes.labelsize']= 16
	plt.plot(fpr, tpr, label='ROC (Área = %0.2f)' % roc_auc,color='magenta')
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'b--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('Taxa de Verdadeiros Positivos')
	plt.xlabel('Taxa de Falsos Positivos')
	plt.grid(True)
	plt.show()
	g.savefig("ROC3.pdf",bbox_inches='tight')
	
	#Exportando para o matlab
	df_fpr = pd.DataFrame(fpr[1:])
	csvFile = open("fpr_roc3.csv", 'w' ,encoding='utf-8')
	df_fpr.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

	df_tpr = pd.DataFrame(tpr[1:])
	csvFile = open("tpr_roc3.csv", 'w' ,encoding='utf-8')
	df_tpr.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average=None,)))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average=None)))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))


if __name__ == "__main__":

	lsa_cluster_oneclass()
