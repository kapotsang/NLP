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

englishLang = set(nltk.corpus.words.words())
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('I')
stopwords.append('u')
tokenizer = RegexpTokenizer(r'\w+')

col_list =["User Review","Star Rating","Date"]
df = pd.read_csv("hubspotCombined.csv", usecols=col_list, encoding='utf-8')
comment = df['User Review'].tolist()
stringComment = ' '.join(map(str,comment))
lowerComment = stringComment.lower()
# sent = " ".join(w for w in nltk.wordpunct_tokenize(lowerComment)
#                 if w.lower() in englishLang or not w.isalpha())
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






