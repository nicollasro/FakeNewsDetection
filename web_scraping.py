# encoding: utf-8
# -*- coding: cp1252 -*-.
#!/usr/bin/python
from time import time, sleep
import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_tweets(tweet):
	stop_words = set(stopwords.words('portuguese'))
	word_tokens = word_tokenize(tweet)

	#after tweepy preprocessing the colon left remain after removing mentions
	#or RT sign in the beginning of the tweet
	tweet = re.sub(r':', '', tweet)
	tweet = re.sub(r'‚Ä¶', '', tweet)
	#replace consecutive non-ASCII characters with a space
	tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet) 


	#remove emojis from tweet
	tweet = emoji_pattern.sub(r'', tweet)

	#filter using NLTK library append it to a string
	filtered_tweet = [w for w in word_tokens if not w in stop_words]
	filtered_tweet = []

	#looping through conditions
	for w in word_tokens:
		#check tokens against stop words , emoticons and punctuations
		if w not in stop_words and w not in emoticons and w not in string.punctuation:
			filtered_tweet.append(w)
	return ' '.join(filtered_tweet)


def write_tweets(account, file):

	# If the file exists, then read the existing data from the CSV file.
	if os.path.exists(file):
		df = pd.read_csv(file,header=0)
		header=True
	else:
		df = pd.DataFrame(columns=COLS)
		header=False
	#page attribute in tweepy.cursor and iteration
	for status in tweepy.Cursor(api.user_timeline, screen_name=account, include_rts=False, lang="pt", tweet_mode="extended").items():
		status = status._json
		new_entry=[]

		#when run the code, below code replaces the retweet amount and
		#no of favorires that are changed since last download.
		if status['created_at'] in df['created_at'].values:
			pass
			
		else:
			#tweepy preprocessing called for basic preprocessing
			clean_text = p.clean(status['full_text'])

			#call clean_tweet method for extra preprocessing
			filtered_tweet=clean_tweets(clean_text)

			#pass textBlob method for sentiment calculations
			blob = TextBlob(filtered_tweet)
			#print (blob.sentences)
			Sentiment = blob.sentiment

			#seperate polarity and subjectivity in to two variables
			polarity = Sentiment.polarity
			subjectivity = Sentiment.subjectivity

			#new entry append
			new_entry += [status['id'], status['created_at'],
					status['source'], status['full_text'],filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
					status['favorite_count'], status['retweet_count']]


			#to append original author of the tweet
			new_entry.append(status['user']['screen_name'])

			try:
				is_sensitive = status['possibly_sensitive']
			except KeyError:
				is_sensitive = None
			new_entry.append(is_sensitive)

			# hashtagas and mentiones are saved using comma separted
			hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
			new_entry.append(hashtags)
			mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
			new_entry.append(mentions)

			#get location of the tweet if possible
			try:
				location = status['user']['location']
			except TypeError:
				location = ''
			new_entry.append(location)

			try:
				coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
			except TypeError:
				coordinates = None
			new_entry.append(coordinates)

			if header:
				single_tweet_df = pd.DataFrame([new_entry])
			else:
				single_tweet_df = pd.DataFrame([new_entry],columns=COLS)

			single_tweet_df.columns=df.columns
			df = pd.concat([df, single_tweet_df])

	csvFile = open(file, 'w' ,encoding='utf-8')
	df.to_csv(csvFile, mode='w', columns=COLS, index=False, encoding="utf-8")
	
def check_limits(api):

	limits=api.rate_limit_status()
	for i in limits['resources'].keys():
		for k in limits['resources'][i].keys(): 
			if limits['resources'][i][k]['limit']!=limits['resources'][i][k]['remaining']:
				reset_time=float(limits['resources'][i][k]['reset'])-time()
				remain=int(limits['resources'][i][k]['limit'])-int(limits['resources'][i][k]['remaining'])
				max_limit=limits['resources'][i][k]['limit']
				print('---- Status com Limite Modificado---------')
				print (limits['resources'][i])
				print("Falta "+str(reset_time/60)+' min para o próximo reset')
				print('Capacidade comprometida '+str(int(remain)/int(max_limit)))

def call(account, file_path, api):
	cont=0
	try:
		cont=cont+1
		write_tweets(account, file_path)
	except:
		print("Tentativa "+str(cont))
		sleep(300)
		call(account, file_path, api)

if __name__ == "__main__":

	#Credenciais da aplicação do Twitter
	consumer_key = 'fjxtMfPPXlEM98Lcr1rCdyUhF'
	consumer_secret = 'D06oj3PIBHzpseZrMattvZtqOsfWEvSkWQzxLjyghD4XpVlb3S'
	access_key= '1173637793379160067-WjK3ITL1xAaphi09UwrdioZn778V8M'
	access_secret = 'YV9zitEUQEC7icCDMrsgphfanwQKi0xOheQPugr7LENy1'

	#Passando as credenciais para o tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)

	#Colunas do arquivo
	COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries']

	#Emotions Felizes
	emoticons_happy = set([':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
	':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
	'=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
	'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)','<3'])

	#Emotions Tristes
	emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', 
	':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('])

	#Padrões de Emojs
	emoji_pattern = re.compile("["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
	u"\U00002702-\U000027B0"
	u"\U000024C2-\U0001F251"
	"]+", flags=re.UNICODE)

	#combine sad and happy emoticons
	emoticons = emoticons_happy.union(emoticons_sad)
	
	#declare keywords as a query for three categories
	account = '@Boatosorg'

	#Nome e caminho do arquivo
	file_path ="boatos.csv"

	#number of posts 
	batch=10

	#call main method passing keywords and file path
	call(account, file_path, api)
	print('Finalizou com sucesso!')
	




