import copy

class PreProcess:
  """
  Class for pre-processing the review before passing it for analyzing
  """
  def __init__(self, data):
    self.data = data

  def tokenize(self):
    from nltk import word_tokenize
    for d in self.data:
      d[1] = word_tokenize(d[1])
    return self.data

  def remove_stopwords(self):
    from nltk.corpus import stopwords
    import re
    stop = set(stopwords.words("english"))
    for d in self.data:
      temp = []
      for w in d[1]:
        # if not a stop word or a piece of punctuation
        if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w):
          temp.append(w)
      d[1] = temp
    return self.data

  def stemmingPS(self):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    temp = copy.deepcopy(self.data)
    for i in range(len(temp)):
        for j in range(len(temp[i][1])):
            temp[i][1][j] = ps.stem(temp[i][1][j])
    return temp

  def stemmingLS(self):
    from nltk.stem import LancasterStemmer
    ls = LancasterStemmer()
    temp = copy.deepcopy(self.data)
    for i in range(len(temp)):
        for j in range(len(temp[i][1])):
            temp[i][1][j] = ls.stem(temp[i][1][j])
    return temp

  def stemmingSB(self):
    from nltk.stem import SnowballStemmer
    sb = SnowballStemmer("english")
    temp = copy.deepcopy(self.data)
    for i in range(len(temp)):
        for j in range(len(temp[i][1])):
            temp[i][1][j] = sb.stem(temp[i][1][j])
    return temp

  def get_pos(self, word):
      from collections import Counter
      from nltk.corpus import wordnet  # To get words in dictionary with their parts of speech

      w_synsets = wordnet.synsets(word)

      pos_counts = Counter()
      pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
      pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
      pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
      pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

      most_common_pos_list = pos_counts.most_common(3)
      return most_common_pos_list[0][0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )

  def lemmatize(self):
      from nltk.stem import WordNetLemmatizer  # lemmatizes word based on it's parts of speech

      wnl = WordNetLemmatizer()
      temp = copy.deepcopy(self.data)
      for i in range(len(self.data)):
          for j in range(len(self.data[i][1])):
              self.data[i][1][j] = wnl.lemmatize(self.data[i][1][j], pos=self.get_pos(self.data[i][1][j]))
      return self.data
