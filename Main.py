from DataSetReader import DataSetReader
from PreProcess import PreProcess
from Vectorizer import Vectorizer
from Classify import Classify
from NewClassifier import Classifier

dsr = DataSetReader(directory="./aclImdb/")

tr_data = dsr.labelled_string_data('train')
# Data split by label
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

# small set of data of size 1000
tr_small = tr_negative[:500]+tr_positive[:500]


train_prp = PreProcess(tr_small)
tr_small = train_prp.tokenize()
train_prp.remove_stopwords()
#train_StemPS = train_prp.stemmingPS()
#train_StemLS = train_prp.stemmingLS()
#train_StemSB = train_prp.stemmingSB()
train_StemLemmatize = train_prp.lemmatize()

vectorizer = Vectorizer(type='fastext', fit_data=tr_small, params={'min_count': 1})
tr_small_vecs = vectorizer.vectorize(tr_small)

tst_data = dsr.labelled_string_data('test')

tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]

tst_small = tst_negative[:500]+tst_positive[:500]

tst_prp = PreProcess(tst_small)
tst_small = tst_prp.tokenize()
tst_prp.remove_stopwords()
tst_StemPS = tst_prp.stemmingPS()
tst_StemLS = tst_prp.stemmingLS()
tst_StemSB = tst_prp.stemmingSB()
tst_StemLemmatize = tst_prp.lemmatize()

tst_small_vecs = vectorizer.vectorize(tst_small)


clf = Classifier('SVC', [d[1] for d in tr_small_vecs], [d[2] for d in tr_small_vecs])
print(clf.tune(
  {
    'C': [0.2, 1.0, 2.0, 3.0, 10.0, 100.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
  },
  [d[1] for d in tst_small_vecs],
  [d[2] for d in tst_small_vecs],
  max_only=False
  )
)

clf = Classifier('KNN', [d[1] for d in tr_small_vecs], [d[2] for d in tr_small_vecs])
print(clf.tune(
  {
    'n_neighbors': [i for i in range(1, 11, 2)]
  },
  [d[1] for d in tst_small_vecs],
  [d[2] for d in tst_small_vecs],
  max_only=False
  )
)



# clf = Classify(tr_small_vecs,tst_small_vecs)

# score= clf.DecisionTrees()
# print(score)

# print(clf.RandomForrests(criterion='entropy'))

# print(clf.SVM())

# print(clf.MLP())

# print(clf.LogisticRegression())