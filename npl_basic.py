import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish')) 

def clean_word(text):
  result = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())
  result = re.sub(r'[^a-zA-Z0-9\s]', '', result)
  return result

def steam(text):
    return text
      
def get_words(tweetList, inSet=True):
  words_in_tweet = [tweet.lower().split() for tweet in tweetList]
  if inSet:
    words_in_tweet = set([clean_word(item) for element in words_in_tweet for item in element])
  else:
    words_in_tweet = [item for element in words_in_tweet for item in element]
  words_in_tweet = [w for w in words_in_tweet if not w in stop_words] 
  return words_in_tweet

def get_frecuencies(tweetList, vocabulary):
    words_in_tweet = get_words(tweetList, False)
    frecuencies = []
    for i in vocabulary:
      frecuencies.append(words_in_tweet.count(i))
    return np.array(frecuencies)

def get_frecuency_table(positive_tw, negative_tw):
    all_tweets = positive_tw + negative_tw
    all_words = list(get_words(all_tweets))
    positive_frecuencies = get_frecuencies(positive_tw, all_words)
    negative_frecuencies = get_frecuencies(negative_tw, all_words)
    return pd.DataFrame(list(zip(all_words,positive_frecuencies,negative_frecuencies)), columns=['words','pos_frec','neg_frec'])

def encode_tweet(frecuency_table, tweet):
  words = get_words([tweet])
  pos_frec = 0
  neg_frec = 0
  for word in words:
    try:
      pos_frec = pos_frec + frecuency_table[frecuency_table['words']==word]["pos_frec"].values[0]
      neg_frec = neg_frec + frecuency_table[frecuency_table['words']==word]["neg_frec"].values[0]
    except:
      print("palabra nueva, no se suma...")  
  return [1, pos_frec, neg_frec]

positive_tw = ['Me siento muy contento', 'Estan muy buenas las casas', 'Muy buena la atencion, estoy muy contento.',
'Que rica esta la comida', 'Excelente servicio']
negative_tw = ['Muy mala atencion', 'las cosas son feas', 'pesima atencion']
all_tweets = positive_tw + negative_tw
frecuency_table = get_frecuency_table(positive_tw, negative_tw)
print(frecuency_table)
x = np.zeros((len(all_tweets),3))
y_positive = np.ones(len(positive_tw))
y_negative = np.zeros(len(negative_tw))
y = np.concatenate((y_positive, y_negative), axis=0)

for i in range(len(all_tweets)):
  x[i:] = encode_tweet(frecuency_table, all_tweets[i])
print(x)
model = LogisticRegression()
model.fit(x,y)
print("Resultado: ",model.intercept_, model.coef_)
test_twwet = "pesima la pizza, mala atencion"
encoded_test_twwet = encode_tweet(frecuency_table, test_twwet)
prediccion = model.predict([encoded_test_twwet])
probability = model.predict_proba([encoded_test_twwet])
print("Prediccion: ",prediccion, probability)