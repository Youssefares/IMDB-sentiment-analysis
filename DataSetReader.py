class DataSetReader:
  """
  Class for operations on review data files stored in file system
  """
  def __init__(self, directory='.', positive=True, negative=True, unsup=False):
    self.directory = directory
    self.positive = positive
    self.negative = negative
    self.unsup = unsup


  def labelled_string_data(self, type):
    """
     Type: 'train' or 'test'
     Returns array of [id, review_string, label]
    """
    data = []

    # Error checking
    if type not in ['train', 'test']:
      raise Exception('expected string: train or test')
    
    # Reading data
    import os, re
    dirs = [
      self.directory+type+'/pos/',
      self.directory+type+'/neg/'
    ]

    for dir in dirs:
      for file in os.listdir(dir):
        file_data = re.match(
          r"(?P<id>\d+)_(?P<label>\d+).txt", str(file)
        ).groupdict()
        data.append([
          int(file_data['id']),
          self.string_from_file(dir+file), 
          int(int(file_data['label']) < 5)
        ])
    return sorted(data, key= lambda x: x[2]*10**8 +x[0])


  def string_from_file(self, file):
    with open(file) as f:
      review = f.read()
    return review


  def tokenize(self, data):
    from nltk import word_tokenize
    for d in data:
      d[1] = word_tokenize(d[1])
    return data


  def remove_stopwords(self, data):
    from nltk.corpus import stopwords
    noise = ['``','?','!','.',',',"'","''",':',';','/']
    stop = set(stopwords.words("english"))
    for d in data:
      temp = []
      for w in d[1]:
        if w not in stop and w not in noise:
          temp.append(w)
      d[1] = temp
    return data
