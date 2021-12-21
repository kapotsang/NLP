import csv
import os
import nltk
import pandas as pd
import tokenizer as tokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
from nltk.corpus import stopwords
import random
import re
from csv import reader
import matplotlib


stopwords = nltk.corpus.stopwords.words('english')
newStopwords = ['I','u','mobile','banking']
stopwords.append(newStopwords)

col_list =[" COMMENT"]
df = pd.read_csv("TWITTER4OUTPUT.csv", usecols=col_list, encoding='utf-8')
comment = df[' COMMENT'].tolist()
stringComment = ' '.join(map(str,comment))
lowerComment = stringComment.lower()
tokenizer = RegexpTokenizer(r'\w+')
tokenizedComment = tokenizer.tokenize(lowerComment)
filteredComment = [w for w in tokenizedComment if not w.lower() in stopwords]
filteredComment = []
for w in tokenizedComment:
    if w not in stopwords:
        filteredComment.append(w)

taggedComment = nltk.pos_tag(filteredComment)
adjectives = []
verbs = []
for (adj, tag) in taggedComment:
    if tag == 'JJ':
        adjectives.append(adj)

adjectivesString = ' '.join(map(str,adjectives))
tokenizedAdj = tokenizer.tokenize((adjectivesString))


#print(filteredComment)
fd = nltk.FreqDist(filteredComment)
fd.plot(20)