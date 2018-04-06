from DataSetReader import DataSetReader
from PreProcess import PreProcess
from Vectorizer import Vectorizer
import numpy as np

dsr = DataSetReader(directory="./aclImdb/")

tr_data = dsr.labelled_string_data('train')

prp = PreProcess(tr_data)
print(tr_data[0])
tr_data = prp.tokenize()
print(tr_data[0])
prp.remove_stopwords()
print(tr_data[0])

# Data split by label
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

# small set of data of size 1000
tr_small = tr_negative[:500]+tr_positive[:500]


vectorizer = Vectorizer(type='count', params={'ngram_range': (1, 3)})

tr_small_vecs = vectorizer.vectorize(tr_small)

tst_data = dsr.labelled_string_data('test')
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]
