import pandas as pd
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words.add('cex')
nlp.Defaults.stop_words.add("'\n'")
nlp.vocab['cex'].is_stop = True

#Get review body as string
col_list =["COMMENT","RATING","TIME"]
df = pd.read_csv("reviews_ioCombined.csv", usecols=col_list, encoding='utf-8')
comment = df['COMMENT'].tolist()
stringComment = ' '.join(map(str,comment))
stringComment1 = nlp(stringComment)



# tokenize review
token_list = []
for token in stringComment1:
#   print(token.text, '\t', token.pos_ ,'\t' ,token.tag_ ,'\t' , token.lemma_)
    token_list.append(token.lemma_)
#    print(f"{token.text:{10}} {token.pos_:{10}}")


# #extract entities and explain labels
# for entity in stringComment1.ents:
#     print(entity)
#     print(entity.label_)
#     print(str(spacy.explain(entity.label_)))
#     print('\n')


filtered_sentence =[]
for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word)

noPunc = []
for word1 in filtered_sentence:
    cleanedSentence = nlp.vocab[word1]
    if cleanedSentence.is_alpha == True:
        noPunc.append(word1)

noPuncString = ' '.join(noPunc)
noPuncString1 = nlp(noPuncString)
for word3 in noPuncString1:
    print(f"{word3.text} \t {word3.pos_} \t {word3.tag_}")

# for sent in stringComment1.sents:
#     print(sent)