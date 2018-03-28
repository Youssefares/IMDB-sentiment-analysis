from DataSetReader import DataSetReader

dsr = DataSetReader(directory="./aclImdb/")
data = dsr.labelled_string_data('train')
negative, positive = data[:len(data)//2], data[len(data)//2:]
