import matplotlib_inline
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)
col_list =["COMMENT","RATING"]
df = pd.read_csv("reviews_ioCombined.csv",encoding='utf-8')
df['LENGTH'] = df['COMMENT'].str.len()
#print(df.isnull().sum())
print(len(df))
#print(df['RATING'].value_counts())
#print(df.head())
commentRaw = df['COMMENT'].tolist()
rating = df['RATING'].tolist()
stringComment = ' '.join(map(str,commentRaw)).lower()
#stringRating = ' '.join(map(str,rating))
stringComment1 = nlp(stringComment)

token_list = []
for token in stringComment1:
    token_list.append(token.lemma_)

stopwordFiltered = []

for words in token_list:
    lexeme = nlp.vocab[words]
    if lexeme.is_stop == False:
        stopwordFiltered.append(words)



noPunc = []
for word1 in stopwordFiltered:
    cleanedSentence = nlp.vocab[word1]
    if cleanedSentence.is_alpha == True:
        noPunc.append(word1)


set1 = noPunc[:len(noPunc)//2]
set2 = noPunc[len(noPunc)//2:]

print(len(set1))
print(len(set2))

set1List = ' '.join(map(str,set1)).lower().split()
set2List = ' '.join(map(str,set2)).lower().split()
# set1Token = nlp(set1List)
# set2Token = nlp(set2List)

# print(set1Token)

vocab = {}
i = 1
for word in set1List:
    if word in vocab:
        continue
    else:
        vocab[word]=i
        i+=1

for word in set2List:
    if word in vocab:
        continue
    else:
        vocab[word] = i
        i+=1

one = ['set1']+[0]*len(vocab)
for word in set1List:
    one[vocab[word]]+=1

two = ['set2']+[0]*len(vocab)
for word in set2List:
    two[vocab[word]]+=1

print(one)
print(two)
print(set1List[4])




# one = ['COMMENT']+[0]*len(vocab)
#
# for word in stringComment1:
#     one[vocab[word]]+=1
#
# print(one)





# matplotlib_inline
# plt.xscale('log')
# bins = 1.15**(np.arange(0,30))
# plt.hist(df[df['RATING']=='5']['LENGTH'],bins=bins, alpha=0.8)
# plt.hist(df[df['RATING']=='4']['LENGTH'],bins=bins, alpha=0.8)
# plt.hist(df[df['RATING']=='3']['LENGTH'],bins=bins, alpha=0.8)
# plt.hist(df[df['RATING']=='2']['LENGTH'],bins=bins, alpha=0.8)
# plt.hist(df[df['RATING']=='1']['LENGTH'],bins=bins, alpha=0.8)
# plt.legend(('5','4','3','2','1'))
# plt.show()

# from sklearn.model_selection import train_test_split
# X = df[['LENGTH']]
# y = df['RATING']
#
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# print(X_train.shape)
# print(X_test)
# print(y_test)

# from sklearn.linear_model import LogisticRegression
# lr_model =  LogisticRegression(solver='lbfgs')
# lr_model.fit(X_train,y_train)
# from sklearn import metrics
# predictions = lr_model.predict(X_test)
# print(metrics.confusion_matrix(y_test, predictions))
# # confusion matrix