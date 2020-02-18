# -*- coding: utf-8 -*-


from pythainlp import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np


df = pd.read_csv("lyrics.csv",sep ="\t")
lyrics = list(df["Lyrics"])

word_toks = []
for song in lyrics:
    word_toks.append(word_tokenize(song , keep_whitespace=False))
    
uniword = list(sorted(set([word for wordlist in word_toks for word in wordlist])))


inverted_index = []
for sen in word_toks:
  d = [True if x in sen else False for x in uniword]
  inverted_index.append(d)

df = pd.DataFrame(np.array(inverted_index).T, columns= list(df["Song name"]))
df["word"] =  uniword 
df = df.set_index("word")
df = df.loc[:,~df.columns.duplicated()]
##query input
howru = input("How are you today?")




def search(query, invidx):
    words = invidx.index
    query_token = word_tokenize(query)
    query_vector = [True if word in query_token else False for word in words]
    
#    docs = []
#    for doc in invidx.columns:       
#        if any( query_vector & invidx[doc].values):
#            docs.append(doc)
        
    docs = [doc if any(np.array(query_vector) & invidx[doc].values) else None for doc in invidx.columns]
    docs = list(filter(None,docs))

    return(docs)

print("Your song is ",search(howru,df))

 
