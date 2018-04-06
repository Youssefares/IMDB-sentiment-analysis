from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Vectorizer:
  def __init__(self, type='count', params={}):
    if type == 'count':
      self.vectorizer = CountVectorizer(**params)
      self.vectorize_call = lambda data: self.vectorizer.fit_transform(data).toarray()

  def vectorize(self, data):
    # extract reviews from second column
    reviews = [' '.join(d[1]) for d in data]
    reviews_vectors = self.vectorize_call(reviews)
    return [[d[0], reviews_vectors[i], d[2]] for i, d in enumerate(data)]
