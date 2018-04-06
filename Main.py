from DataSetReader import DataSetReader
from PreProcess import PreProcess

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
dummy = prp.stemmingPS()
print('Stemming data using Porter')
print(dummy[0])
print()
dummy = prp.lemmatize()
print('Lemmatization data')
print(dummy[0])
print()
prp.stemmingLS()
print('Stemming data using Lancaster')
print(tr_data[0])
print()
prp.stemmingSB()
print('Stemming data using Snowball')
print(tr_data[0])
print()
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

tst_data = dsr.labelled_string_data('test')
prp_tst = PreProcess(tst_data)
print(tst_data[0])
tst_data = prp_tst.tokenize()
print(tst_data[0])
prp_tst.remove_stopwords()
print(tst_data[0])
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]



