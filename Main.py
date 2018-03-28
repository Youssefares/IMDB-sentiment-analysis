from DataSetReader import DataSetReader

dsr = DataSetReader(directory="./aclImdb/")

tr_data = dsr.labelled_string_data('train')
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

tst_data = dsr.labelled_string_data('test')
tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]


print(tst_data[-1])