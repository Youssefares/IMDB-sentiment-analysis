import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec, FastText
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorizer:
  def __init__(self, fit_data, type='count', params={}):
    if type == 'count':
      self.vectorizer = CountVectorizer(**params)
      self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
      self.vectorize_call = lambda data: self.vectorizer.transform(data).toarray()
    
    elif type == 'tfidf':
      self.vectorizer = TfidfVectorizer(**params)
      self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
      self.vectorize_call = lambda data: self.vectorizer.transform(data).toarray()

    elif type == 'wordembedd':
      sentences = [d[1] for d in fit_data]
      self.vectorizer = Word2Vec(sentences, **params)
      self.vectorizer = self.vectorizer.wv
      min = np.min([len(sentence) for sentence in sentences])
      self.vectorize_call = lambda data: [
        np.array([self.vectorizer[word] for word in sentence[:min] if word in self.vectorizer.vocab]).flatten() for sentence in sentences
      ]
    elif type == 'fastext':
      sentences = [d[1] for d in fit_data]
      self.vectorizer = FastText(sentences, **params)
      self.vectorizer = self.vectorizer.wv
      min = np.min([len(sentence) for sentence in sentences])
      self.vectorize_call = lambda data: [
        np.array([self.vectorizer[word] for word in sentence[:min] if word in self.vectorizer.vocab]).flatten() for
        sentence in sentences
      ]


  def vectorize(self, data):
    # extract reviews from second column
    reviews = [' '.join(d[1]) for d in data]
    reviews_vectors = self.vectorize_call(reviews)
    return [[d[0], reviews_vectors[i], d[2]] for i, d in enumerate(data)]
