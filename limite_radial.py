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
from math import sqrt
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
from sklearn.metrics import silhouette_score,roc_curve,auc, roc_auc_score
from sklearn.preprocessing import normalize
from scipy.stats import ttest_1samp,ttest_ind, ttest_rel, norm

def esfera ():
        df = pd.read_csv("full_database_cleaned.csv",header=0)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
        tfidf_matrix=tfidf_vectorizer.fit_transform(df["clean_text"].values.astype('U'))

        svd = TruncatedSVD(n_components=2000)
        tfidf_lsa = svd.fit_transform(tfidf_matrix)
        tfidf_lsa=normalize(tfidf_lsa.T)
        tfidf_lsa=tfidf_lsa.T

        df_x = pd.DataFrame(tfidf_lsa[0:,0:])
        df2=df_x*df_x
        df2.loc[:,'Total'] = df2.sum(axis=1)
        media_verdadeira=np.mean(df2['Total'].loc[df["probability"] == 1])
        media_falsa=np.mean(df2['Total'].loc[df["probability"] == -1])
        
        desvio_verdadeiro=np.std(df2['Total'].loc[df["probability"] == 1])
        desvio_falso=np.std(df2['Total'].loc[df["probability"] == -1])

        variancia_verdadeira= desvio_verdadeiro**2
        variancia_falsa= desvio_falso**2

        intervalo_verdadeiro=(1.96*(desvio_verdadeiro/sqrt(30809)))
        intervalo_falso=(1.96*(desvio_falso/sqrt(3180)))

        intervalo_ver_pos=intervalo_verdadeiro+media_verdadeira
        intervalo_ver_neg=media_verdadeira-intervalo_verdadeiro
        
        TN=TP=FP=FN=0

        f=plt.figure(3)
        plt.rcParams['axes.labelsize']= 16
        n, bins, patches = plt.hist(df2['Total'].loc[df["probability"] == 1],density=True,label='Notícias Legítmas', alpha=0.5, bins= int(sqrt(30809)))
        n, bins, patches = plt.hist(df2['Total'].loc[df["probability"] == -1],density=True, label='Notícias Falsas', alpha=0.5, bins=int(sqrt(3180)))
        plt.ylabel('P(X = x)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

        probabilidades=[]
        edges=np.histogram_bin_edges(df2['Total'].loc[df["probability"] == 1], bins= int(sqrt(30809))) #Retorna uma lista com o limite sup e inf de cada bin
        for index, row in df2.iterrows():
                bin_correspondente=np.digitize(row['Total'],bins) #Acha o bin correspondente
                r_menos=edges[bin_correspondente] #Acha o limite inferior
                r_mais=edges[bin_correspondente-1] #Acha o limite superior
                z_menos=(r_menos-media_verdadeira)/desvio_verdadeiro #Acha o z_menos
                z_mais=(r_mais-media_verdadeira)/desvio_verdadeiro #Acha o z_mais
                prob_menos=norm.cdf(z_menos)-0.5
                prob_mais=norm.cdf(z_mais)-0.5
                #print (prob_menos)
                #print (prob_mais)
                prob_final=prob_mais-prob_menos
                probabilidades.append(prob_final)
                #print('Probabilidade Final'+str(prob_final))
                if prob_final>=0.5:
                        if df['probability'][index]==1:
                                TP=TP+1
                        else:
                                FP=FP+1
                else:
                        if df['probability'][index]==-1:
                                TN=TN+1
                        else:
                                FN=FN+1
        print ('TP: '+str(TP))
        print ('FP: '+str(FP))
        print ('TN: '+str(TN))
        print ('FN: '+str(FN))
        precisao=TP/(TP+FP)
        sensibilidade=TP/(TP+FN)
        acuracia=(TP+TN)/(TN+TP+FP+FN)
        print ('Acuracia :'+str(acuracia))
        print ('sensibilidade :'+str(sensibilidade))
        print ('Precisao :'+str(precisao))
        
        df2.loc[df["probability"] != 1, "True"] = 0
        df2.loc[df["probability"] == 1, "True"] = 1

        true=[]
        print (len(probabilidades))
        for i in range(len(probabilidades)):
                probabilidades[i]=1-probabilidades[i]
        for index, row in df.iterrows():
                true.append(row['probability'])
        print (len(true))
        fpr, tpr, thresholds = roc_curve(true, probabilidades)
        roc_auc = auc(fpr, tpr)
        g=plt.figure(3)
        plt.plot(fpr, tpr, label='ROC (Área = %0.2f)' % roc_auc,color='orange')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'b--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.grid(True)
        plt.show()
        #g.savefig("ROC.pdf",bbox_inches='tight')

        #Exportando para o matlab
        df_fpr = pd.DataFrame(fpr[1:])
        csvFile = open("fpr_roc.csv", 'w' ,encoding='utf-8')
        df_fpr.to_csv(csvFile, mode='w', index=False, encoding="utf-8")

        df_tpr = pd.DataFrame(tpr[1:])
        csvFile = open("tpr_roc.csv", 'w' ,encoding='utf-8')
        df_tpr.to_csv(csvFile, mode='w', index=False, encoding="utf-8")
        
if __name__ == "__main__":

	esfera()
