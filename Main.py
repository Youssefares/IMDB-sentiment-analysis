from DataSetReader import DataSetReader
from PreProcess import PreProcess

dsr = DataSetReader(directory="./aclImdb/")

tr_data = dsr.labelled_string_data('train')
prp = PreProcess(tr_data)

tkn_tr_data = prp.tokenize(tr_data)
prp.remove_stopwords(tkn_tr_data)
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

tst_data = dsr.labelled_string_data('test')
tkn_tst_data = prp.tokenize(tst_data)
prp.remove_stopwords(tkn_tst_data)
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]



