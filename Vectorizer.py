import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec, FastText
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorizer:
  def __init__(self, fit_data, tst_data, type='count', params={}):
    self.fit_data = fit_data
    self.tst_data = tst_data
    if type == 'count':
      self.vectorizer = CountVectorizer(**params)
      self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
      self.tst_vecs = self.vectorizer.transform([' '.join(d[1]) for d in self.tst_data]).toarray()
      self.fit_vecs = self.vectorizer.transform([' '.join(d[1]) for d in self.fit_data]).toarray()

    elif type == 'tfidf':
      self.vectorizer = TfidfVectorizer(**params)
      self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
      self.tst_vecs = self.vectorizer.transform([' '.join(d[1]) for d in self.tst_data]).toarray()
      self.fit_vecs = self.vectorizer.transform([' '.join(d[1]) for d in self.fit_data]).toarray()


    elif type == 'wordembedd':
      fit_sentences = [d[1] for d in fit_data]
      tst_sentences = [d[1] for d in tst_data]
      self.vectorizer = Word2Vec(fit_sentences, **params)
      self.vectorizer = self.vectorizer.wv
      self.fit_vecs = [
        np.array([self.vectorizer[word] for word in sentence if word in self.vectorizer.vocab]).flatten() for sentence in fit_sentences
      ]
      
      self.tst_vecs = [
        np.array([self.vectorizer[word] for word in sentence if word in self.vectorizer.vocab]).flatten() for sentence in tst_sentences
      ]

      self.min = min(
        np.min([len(vector) for vector in self.fit_vecs]),
        np.min([len(vector) for vector in self.tst_vecs])
      )

      self.fit_vecs = [fit_vec[:self.min] for fit_vec in self.fit_vecs]
      self.tst_vecs = [tst_vec[:self.min] for tst_vec in self.tst_vecs]

    elif type == 'fasttext':
      fit_sentences = [d[1] for d in fit_data]
      tst_sentences = [d[1] for d in tst_data]
      self.vectorizer = FastText(fit_sentences, **params)
      self.vectorizer = self.vectorizer.wv
      self.fit_vecs = [
        np.array([self.vectorizer[word] for word in sentence if word in self.vectorizer.vocab]).flatten() for sentence in fit_sentences
      ]
      
      self.tst_vecs = [
        np.array([self.vectorizer[word] for word in sentence if word in self.vectorizer.vocab]).flatten() for sentence in tst_sentences
      ]

      self.min = min(
        np.min([len(vector) for vector in self.fit_vecs]),
        np.min([len(vector) for vector in self.tst_vecs])
      )

      self.fit_vecs = self.fit_vecs[:self.min]
      self.tst_vecs = self.tst_vecs[:self.min]


    # if type == 'count':
    #   self.vectorizer = CountVectorizer(**params)
    #   self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
    #   self.vectorize_call = lambda data: self.vectorizer.transform([' '.join(d) for d in data]).toarray()

    # elif type == 'tfidf':
    #   self.vectorizer = TfidfVectorizer(**params)
    #   self.vectorizer.fit([' '.join(d[1]) for d in fit_data])
    #   self.vectorize_call = lambda data: self.vectorizer.transform([' '.join(d) for d in data]).toarray()

    # elif type == 'wordembedd':
    #   sentences = [d[1] for d in fit_data]
    #   self.vectorizer = Word2Vec(sentences, **params)
    #   self.vectorizer = self.vectorizer.wv
    #   self.vectorize_call = lambda data: self.vectorize_call_helper(data)

    # elif type == 'fasttext':
    #   sentences = [d[1] for d in fit_data]
    #   self.vectorizer = FastText(sentences, **params)
    #   self.vectorizer = self.vectorizer.wv
    #   self.vectorize_call = lambda data: self.vectorize_call_helper(data)

  def vectorize(self):
    return ([[d[0], self.fit_vecs[i], d[2]] for i, d in enumerate(self.fit_data)],
    [[d[0], self.tst_vecs[i], d[2]] for i, d in enumerate(self.tst_data)])


  # def vectorize_call_helper(self, min):
  #   non_equal_vectors = [
  #     np.array([self.vectorizer[word] for word in sentence if word in self.vectorizer.vocab]).flatten() for sentence in data
  #   ]
  #   min_len = self.min or min([len(vector) for vector in non_equal_vectors])
  #   self.min = min_len
  #   print(self.min)
  #   equal_vectors = np.array([vector[:min_len] for vector in non_equal_vectors])
  #   for i, vector in enumerate(equal_vectors):
  #     assert i < 1 or len(vector) == len(equal_vectors[i-1])
  #   return equal_vectors