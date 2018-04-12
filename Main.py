from DataSetReader import DataSetReader
from PreProcess import PreProcess
from Vectorizer import Vectorizer
from Classify import Classify
from NewClassifier import Classifier
import itertools
import numpy as np
import matplotlib.pyplot as plt

dsr = DataSetReader(directory="../aclImdb/")

tr_data = dsr.labelled_string_data('train')
# Data split by label
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

# small set of data of size 1000
tr_small = tr_negative[:500]+tr_positive[:500]


tst_data = dsr.labelled_string_data('test')

tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]

tst_small = tst_negative[:500]+tst_positive[:500]


# Preprocessing combinations
cleaning_operations = ['remove_stopwords','lemmatize','stemmingLS','stemmingPS','stemmingSB'];
combinations = [i  for j in range(len(cleaning_operations)) for i in itertools.combinations(cleaning_operations,j+1)]

# Vectorization combinations
vec_list = [['tfidf',{}],['count',{}],['wordembedd',{}],['fasttext',{}]]

# Classifier combinations
clf_list = [['KNN',{}],['SVC',{}],['DecisionTree',{}],['RandomForestClassifier',{}],['LogisticRegression',{}]
  ,['MLP',{}],['AdaBoost',{}],['Bagging',{}]]

clf_list = [['KNN',{'n_neighbors': [i for i in range(1, 4, 2)]}]]
vec_list = [['tfidf',{}]]


# ex: {'KNN': [(acc, vectorization, preprocessing_ops)], 'LR': ... }
clf_dict = {clf[0]: {} for clf in clf_list}
for vec in vec_list:
  print('Vectorization technique: '+vec[0].__str__())
  for combination in combinations:
    print('Preprocessing : ' + combination.__str__())
    tr_prp = PreProcess(tr_small)
    tst_prp = PreProcess(tst_small)

    tr_clean_data = tr_prp.clean(combination)
    tst_clean_data = tst_prp.clean(combination)

    vectorizer = Vectorizer(type=vec[0],fit_data=tr_clean_data,params=vec[1])
    tr_small_vecs = vectorizer.vectorize(tr_clean_data)
    tst_small_vecs = vectorizer.vectorize(tst_clean_data)
    for cl in clf_list:
      print('Classifier: ' + cl[0].__str__())

      clf = Classifier(cl[0], [d[1] for d in tr_small_vecs], [d[2] for d in tr_small_vecs])
      params_accs = clf.tune(
        cl[1],
        [d[1] for d in tst_small_vecs],
        [d[2] for d in tst_small_vecs],
        max_only=False
      )
      # should have dict of clfs contatining dict of params with all accuracies & methods tried
      for key, value in params_accs.items():
        if key in clf_dict[cl[0]]:
          clf_dict[cl[0]][key] += [(value, vec, combination)]
        else:
          clf_dict[cl[0]][key] = [(value, vec, combination)]
      


