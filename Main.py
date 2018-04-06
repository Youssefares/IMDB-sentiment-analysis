from DataSetReader import DataSetReader
from PreProcess import PreProcess
from Vectorizer import Vectorizer
import numpy as np

dsr = DataSetReader(directory="./aclImdb/")

tr_data = dsr.labelled_string_data('train')

prp = PreProcess(tr_data)
print('Triainning data')
print(tr_data[0])
print()
tr_data = prp.tokenize()
print('Tokenized data')
print(tr_data[0])
print()
prp.remove_stopwords()
print('Remove stopwords')
print(tr_data[0])
print()
# dummy = prp.stemmingPS()
# print('Stemming data using Porter')
# print(dummy[0])
# print()
# dummy = prp.lemmatize()
# print('Lemmatization data')
# print(dummy[0])
# print()
# dummy = prp.stemmingLS()
# print('Stemming data using Lancaster')
# print(dummy[0])
# print()
# dummy = prp.stemmingSB()
# print('Stemming data using Snowball')
# print(dummy[0])
# print()


# Data split by label
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

# small set of data of size 1000
tr_small = tr_negative[:500]+tr_positive[:500]

vectorizer = Vectorizer(type='tfidf', params={'ngram_range': (1, 3)})
tr_small_vecs = vectorizer.vectorize(tr_small)

tst_data = dsr.labelled_string_data('test')
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]
