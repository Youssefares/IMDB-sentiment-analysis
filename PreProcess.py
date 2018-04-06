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