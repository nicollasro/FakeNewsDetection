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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def remove_punctuation(word):

	word = word.replace('-via', '')
	word = word.replace('(via', '')
	word = word.replace('gt', '')
	for ch in string.punctuation:
		word = word.replace(ch, "")
	if(len(word) < 2 or word.isdigit()):
		return ""
	return word

def similarity(word, names):

	try:
		for w in names[word[0]]:
			if(fuzz.ratio(w, word) > 80):
				return True
	except KeyError:
		pass
	return False

def get_name_dict():

	names = open("names.txt", encoding="utf-8")
	names_lines = names.readlines()
	names_dict = {}
	for lines in names_lines:
		line = lines.replace('\n','').lower()
		if line[0] not in names_dict: names_dict[line[0]] = []
		names_dict[line[0]].append(line)
	return names_dict

def space_vector_model():
#def space_vector_model(df):
	
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	#sns.set()
	#sns.countplot(y="original_author", data=df)
	#plt.show()
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	#tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])
	tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	#print(tfidf_vectorizer.get_feature_names())
	X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df["probability"], test_size=0.1)
	
	## gama grande (apenas samples próximos influênciam) -> curva sinuosa. Já o gama pequeno (samples distantes também influencia) -- curva reta
	clf = OneClassSVM(nu=0.01, kernel="rbf",gamma=0.0001)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)
	#y_pred = [1]*len(y_test)
	
	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average='micro')))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average='micro')))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))

	#c_matrix = confusion_matrix(y_test,y_pred)
	#print (c_matrix)
	
	#sns.set(style="ticks", color_codes=True, rc={"figure.figsize": (12, 8)}, font_scale=1.2)
	#sns.heatmap(c_matrix, annot=True, annot_kws={"size": 2})
	#plt.show()
	#clf.score_samples(X)
def esfera ():
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

	print('LSA...')
	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	tfidf_lsa=normalize(tfidf_lsa)
	
	print("OneClassSVM...")
	df_x = pd.DataFrame(tfidf_lsa[0:,0:])
	print (df_x.shape)
	'''
	X_train, X_test, y_train, y_test = train_test_split(df_x, df['probability'][0:], test_size=0.3)
	
	train_normal=X_train.loc[df["probability"] == 1]
	train_outliers = X_train.loc[df["probability"] == -1]
	outlier_prop = len(train_outliers)/len(df_x)

	clf = OneClassSVM(nu=outlier_prop, kernel="poly",gamma=0.000001, degree=5)
	clf.fit(X_train)
	y_pred = clf.predict(X_test)
	#print(y_test)

	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average=None,)))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average=None)))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))
	'''

def kmeans ():
	df = pd.read_csv("full_database_cleaned_backup.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

	print('LSA...')
	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	#tfidf_lsa=normalize(tfidf_lsa)
	tfidf_lsa=normalize(tfidf_lsa.T)
	tfidf_lsa=tfidf_lsa.T
	df['tfidf_lsa'] = pd.DataFrame(tfidf_lsa[0:,0:])
	
	'''
	pca = PCA(n_components=1900)
	tfidf_pca = pca.fit_transform(tfidf_lsa)
	#print(pca.explained_variance_ratio_) #quanta informação esta comprimida nas primeiras componentes 
	print (pca.explained_variance_ratio_.sum())
	#print("Clustering... KMeans")
	'''
	true_k = 56
	kmeans = KMeans(n_clusters=true_k, init='k-means++', max_iter=10000, n_init=1)
	kmeans.fit(tfidf_lsa)
	
	#print (len(kmeans.cluster_centers_)) #Linhas
	#print (len(kmeans.cluster_centers_[0])) #Colunas
	transf_matrix=kmeans.cluster_centers_.T
	#short_matrix=np.dot(tfidf_lsa,transf_matrix)
	short_matrix=np.matmul(tfidf_lsa,transf_matrix)
	#print (len(short_matrix)) #Linhas
	#print (len(short_matrix[0])) #Colunas
	
	print("OneClassSVM...")
	df_x = pd.DataFrame(short_matrix[0:,0:])
	print (short_matrix.shape)
	print (df_x.shape)
	
	print("Treinamento")

	df_x = pd.DataFrame(tfidf_lsa[0:,0:])
	X_train, X_test, y_train, y_test = train_test_split(df_x, df['probability'][0:], test_size=0.1)
	
	train_normal=X_train.loc[df["probability"] == 1]
	train_outliers = X_train.loc[df["probability"] == -1]
	outlier_prop = len(train_outliers)/len(df_x)
	gama=0.01
	
	clf = OneClassSVM(nu=outlier_prop, kernel="linear",gamma=gama, degree=5)
	#clf.fit(df_x.loc[df["probability"] == 1])
	clf.fit(train_normal)
	y_pred = clf.predict(X_test)
	y_score=[]
	y_score = max(clf.decision_function(X_test)) - clf.decision_function(X_test)
	print(y_score)
	print ("Gama " + str(gama))
	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average=None,)))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average=None)))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))

def kmeans_silhouete():


	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

	print('LSA...')
	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	tfidf_lsa=normalize(tfidf_lsa.T)
	tfidf_lsa=tfidf_lsa.T

	print('Silhouete...')
	num_clusters = np.arange(2,150)
	results = {}

	for size in num_clusters:
		print("testing k = ",size)
		model = KMeans(n_clusters=size, init='k-means++', max_iter=1000).fit(tfidf_lsa)
		predictions = model.predict(tfidf_lsa)
		results[size] = silhouette_score(tfidf_lsa, predictions)

	print(results)
	best_size = max(results, key=results.get)
	print(best_size)

def kmeans_elbow():
	'''
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	#corpora = pd.DataFrame(tfidf_matrix.toarray())
	feature_names=tfidf_vectorizer.get_feature_names()

	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	'''

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

	print('LSA...')
	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	tfidf_lsa=normalize(tfidf_lsa.T)
	tfidf_lsa=tfidf_lsa.T

	print('Elbow...')
	sse = {}
	for k in range(2, 20):
		print("testing k = ",k)
		kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000).fit(tfidf_lsa)
	    #df["clusters"] = kmeans.labels_
	    #print(data["clusters"])
		sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
		maximo=max(sse.values())
	#for key in sse:
		#sse[key] = sse[key]/maximo
	print (sse)
	plt.figure()
	plt.plot(list(sse.keys()), list(sse.values()))
	plt.xlabel("Número de Clusters")
	plt.ylabel("SSE (Normalizado)")
	plt.show()


def teste_LSA():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	#corpora = pd.DataFrame(tfidf_matrix.toarray())
	feature_names=tfidf_vectorizer.get_feature_names()
	print (len(feature_names))

	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	tfidf_lsa=normalize(tfidf_lsa)
	#print(svd.explained_variance_ratio_)
	#print(svd.explained_variance_ratio_.sum()) #tem que ser menor que 70%
	#print (svd.singular_values_)
	#print (tfidf_lsa)
	#principal_comp=pd.DataFrame(svd.components_, columns=feature_names)
	#print (principal_comp)
	#print (type(tfidf_lsa))
	print (len(tfidf_lsa)) 
	print (len(tfidf_lsa[0])) #numero de features
	df_x = pd.DataFrame(tfidf_lsa[1:,1:])
	#print(df_x.shape)
	#print(df['probability'].shape)
	X_train, X_test, y_train, y_test = train_test_split(df_x, df['probability'][1:], test_size=0.1)
	#print (X_train.shape)
	train_normal=X_train.loc[df["probability"] == 1]
	train_outliers = X_train.loc[df["probability"] == -1]
	outlier_prop = len(train_outliers)/len(train_normal)
	#print (outlier_prop)
	clf = OneClassSVM(nu=outlier_prop, kernel="linear",gamma=0.000001)
	clf.fit(train_normal)
	y_pred = clf.predict(X_test)
	#print (y_pred)
	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average='micro')))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average='micro')))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))

def graficos():
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	sns.set()
	sns.countplot(y="original_author", data=df)
	plt.show()

def teste_LSA_puro():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	#corpora = pd.DataFrame(tfidf_matrix.toarray())
	feature_names=tfidf_vectorizer.get_feature_names()
	print (len(feature_names))

	svd = TruncatedSVD(n_components=2000)
	tfidf_lsa = svd.fit_transform(tfidf_matrix)
	#print(svd.explained_variance_ratio_)
	print(svd.explained_variance_ratio_.sum()) #tem que ser menor que 70%
	#print (svd.singular_values_)
	#print (tfidf_lsa)
	#principal_comp=pd.DataFrame(svd.components_, columns=feature_names)
	#print (principal_comp)
	#print (type(tfidf_lsa))
	
def teste_PCA():
	#não funciona com matrix esparsa
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	corpora = pd.DataFrame(tfidf_matrix.toarray())
	feature_names=tfidf_vectorizer.get_feature_names()

	pca = PCA()
	tfidf_pca = pca.fit_transform(tfidf_matrix)
	print(pca.explain_variance_ratio) #quanta informação esta comprimida nas primeiras componentes 
	print (pca.explain_variance_ratio.sum()) #deve estar acima que 70% (70% dos dados originais foram mantidos)

def teste():
#def space_vector_model(df):
	
	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	#print (tfidf_matrix[2])
	print (tfidf_matrix)
	df3 = pd.DataFrame(tfidf_matrix.toarray(), columns = tfidf_vectorizer.get_feature_names())
	df4 = pd.DataFrame(tfidf_matrix.toarray())
	X_train, X_test, y_train, y_test = train_test_split(df4, df['probability'], test_size=0.1)
	print (len(df['probability']))
	print (tfidf_matrix.shape[0])
	print (df4)
	#print (X_train)
	#print (len(X_train))
	#print (df3)
	#train, test = train_test_split(tf, test_size=0.2)
	train_normal=X_train.loc[df["probability"] == 1]
	print (train_normal.shape)
	#print (np.asarray(train_normal["tfidf_matrix"]))
	train_outliers = X_train.loc[df["probability"] == -1]
	print (train_outliers.shape)
	outlier_prop = len(train_outliers)/len(train_normal)
	print (outlier_prop)
	## gama grande (apenas samples próximos influênciam) -> curva sinuosa. Já o gama pequeno (samples distantes também influencia) -- curva reta
	clf = OneClassSVM(nu=outlier_prop, kernel="rbf",gamma=0.000001)
	#print (type(train_normal["tfidf_matrix"].to_frame()))
	clf.fit(train_normal)
	y_pred = clf.predict(X_test)
	#print (y_pred)
	##VERIFICAR O QUE TA SAINDO NO y_pred
	print ("Acurácia: " + str(accuracy_score(y_test,y_pred)))
	print ("Recall: " + str(recall_score(y_test,y_pred,average='micro')))
	print ("Precisão: " + str(precision_score(y_test,y_pred,average='micro')))
	print ("F1-Score: " + str(f1_score(y_test,y_pred)))


def grid_search_multiple_score():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df["probability"], test_size=0.1)

	scores = ['precision', 'recall','accuracy', 'f1']
	nus = [0.0001, 0.001, 0.05, 0.01, 0.1]
	gammas = [0.0001, 0.001, 0.05, 0.01, 0.1]
	tuned_parameters = {'kernel' : ['rbf'], 'gamma' : gammas, 'nu': nus}
	for score in scores:
		clf = GridSearchCV(	estimator=OneClassSVM(), 
							param_grid=tuned_parameters,
							cv=3, 
							scoring=score, 
							return_train_score=True, 
							refit='accuracy')
		clf.fit(X_train, y_train)
		resultDf = pd.DataFrame(clf.cv_results_)
		print(resultDf[["mean_test_score", "std_test_score", "params"]].sort_values(by=["mean_test_score"], ascending=False).head(16))
		print("Best parameters set found on development set:")
		print("Melhor conjunto de parametros: "+str(clf.best_params_))
		print("Melhor Score: "+str(clf.best_score_))

def grid_search_one_score():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df["probability"])

	nus = [0.001, 0.01, 0.1, 1]
	gammas = [0.001, 0.01, 0.1, 1]
	tuned_parameters = {'kernel' : ['rbf'], 'gamma' : gammas, 'nu': nus}
	grid = GridSearchCV(estimator=OneClassSVM(), param_grid=tuned_parameters, cv=5, scoring='accuracy', refit=True)
	grid.fit(X_train, y_train)
	print("Melhor Score: "+ str(grid.best_score_))
	print("Melhor Parametro: "+str(grid.best_params_))
	print("Melhor Estimador: "+str(grid.best_estimator_))

def grid_manually():

	df = pd.read_csv("full_database_cleaned.csv",header=0)
	tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
	tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))
	X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df["probability"])
	nu_values = [0.0001,0.001, 0.05, 0.01, 0.1, 1]
	gamma_values = [0.0001, 0.001, 0.05, 0.01, 0.1, 1]

	best_score=0
	best_parameters={'nu':None, 'gamma':None}
	for nu in nu_values:
		for gamma in gamma_values:
			clf = OneClassSVM(nu=nu, kernel="rbf",gamma=gamma)
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			accuracy= accuracy_score(y_test,y_pred)
			if accuracy > best_score:
				best_score= accuracy
				best_parameters['nu']=nu
				best_parameters['gamma']=gamma
	print (best_parameters)
	print (best_score)

			

def preprocessing(names_dict):
	# REMOVER LINHAS SEM COM O ROTULO
	stemmer = nltk.stem.RSLPStemmer()
	stop_words = set(stopwords.words('portuguese'))
	
	df = pd.read_csv("full_database.csv",header=0)
	df = df.drop(['place','place_coord_boundaries'], axis=1)

	df.loc[df["original_author"] == "Boatosorg", "probability"] = -1
	df.loc[df["original_author"] != "Boatosorg", "probability"] = 1 
	#print (df.shape)
	#df = df.head(2000)
	df["clean_text"] = df["clean_text"].str.lower()
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(remove_punctuation(word) for word in word_tokenize(str(x))))
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(word for word in word_tokenize(x) if word not in stop_words))
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(stemmer.stem(word) for word in word_tokenize(x) if not similarity(word, names_dict)))
	
	COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'probability']
	
	csvFile = open("full_database_cleaned.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")
	
	return df

def cluster_database():
	"""Método para agrupar todos os arquivos .csv da pasta atual em um único arquivo. Cada arquivo do conjunto contém posts de um conta do twitter um veiculo da impressa confiável."""
	
	files = os.listdir('.')
	database = open('full_database.csv','w')
	header = False
	for file_name in files: 
			if ".csv" in file_name:
				parcial_database = open(file_name,"r")
				parcial_database_lines = parcial_database.readlines()
				for line in parcial_database_lines:
					if "created_at" in line and header == False:
						database.write(line)
						header == True
					elif "created_at" in line and header == True:
						print (line)
						continue
					else:
						database.write(line)

if __name__ == "__main__":

	#cluster_database()
	#print (cluster_database.__doc__)

	#names_dict=get_name_dict()

	#df=preprocessing(names_dict)
	#space_vector_model()
	#grid_search_multiple_score()
	#grid_search_one_score()
	#grid_manually()
	#teste()
	#teste_PCA()
	#teste_LSA()
	#teste_LSA_puro()
	#kmeans_silhouete()
	#kmeans_elbow()
	#graficos()
	kmeans()
	#esfera()
