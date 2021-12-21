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
import numpy as np




stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('I')
stopwords.append('u')
col_list =["COMMENT","RATING","TIME"]
df = pd.read_csv("reviews_ioCombined.csv", usecols=col_list, encoding='utf-8')
comment = df['COMMENT'].tolist()
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
posComments = []
for (adj, tag) in taggedComment:
    if tag == 'VB':
        posComments.append(adj)
    if tag == 'JJ':
        posComments.append(adj)
    if tag =='ADV':
        posComments.append(adj)

posCommentString = ' '.join(map(str,posComments))
tokenizedPos = tokenizer.tokenize((posCommentString))
randomPOS = random.shuffle(tokenizedPos)
print(randomPOS)
#print(filteredComment)
fd = nltk.FreqDist(tokenizedPos)
fd.plot(20)




