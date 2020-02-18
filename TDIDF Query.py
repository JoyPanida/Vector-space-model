# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:24:05 2020

@author: USER
"""
# %%

from pythainlp import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
#%%

df = pd.read_csv("lyrics.csv",sep ="\t")
lyrics = list(df["Lyrics"])

# %%
corpus = lyrics
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
tfidf = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()).transpose()
tfidf.columns = list(df["Song name"])
print(tfidf)
# %%

howruu = input("How are you today?")
# query vector
query = howruu
query_vector = vectorizer.transform([query])
q = pd.DataFrame(query_vector.toarray(), columns = vectorizer.get_feature_names()).transpose()
print(q)


# %%
song_ls = tfidf.columns

sim = 1 - cdist(tfidf.values.T, query_vector.toarray(), metric = "cosine")
for i in range(len(sim)):
    if sim[i] == np.max(sim):
        songsug = song_ls[i]
        
print("Your song is:" , songsug)

# %%

#songsug = [ + str(i) if sim[i] == np.max(sim) else None for i in range(len(sim))]
        
#songsug = list(filter(None, songsug)) 
#
#print(q)
#
## result = list(q.columns)
#print("Your song is ", str(songsug))