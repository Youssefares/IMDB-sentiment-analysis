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

vectorizer = Vectorizer(type='count')
tr_data = vectorizer.vectorize(tr_data)
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]



tst_data = dsr.labelled_string_data('test')
# prp_tst = PreProcess(tst_data)
# print(tst_data[0])
# tst_data = prp_tst.tokenize()
# print(tst_data[0])
# prp_tst.remove_stopwords()
# print(tst_data[0])
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]
