# encoding: utf-8
# -*- coding: cp1252 -*-.
#!/usr/bin/python
import nltk
import string
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from time import time, sleep
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

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

def natural_language_processing():

	stemmer = nltk.stem.RSLPStemmer()
	stop_words = set(stopwords.words('portuguese'))
	names_dict=get_name_dict()
	
	df = pd.read_csv("boatos.csv",header=0)

	df.loc[df["user"] == "Boatosorg", "label"] = -1
	df.loc[df["user"] != "Boatosorg", "label"] = 1 
	#print (df.shape)
	#df = df.head(2000)
	df["clean_text"] = df["full_text"].str.lower()
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(remove_punctuation(word) for word in word_tokenize(str(x))))
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(word for word in word_tokenize(x) if word not in stop_words))
	df["clean_text"] = df["clean_text"].apply(lambda x:' '.join(stemmer.stem(word) for word in word_tokenize(x) if not similarity(word, names_dict)))
	
	COLS = ['id', 'created_at', 'source','user','full_text','label']
	
	csvFile = open("database.csv", 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=df.columns, index=False, encoding="utf-8")

if __name__ == "__main__":
	natural_language_processing()
	