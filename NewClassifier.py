from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from itertools import product

class Classifier:
  def __init__(self, clf_strategy, fit_data, fit_labels):
    classifers = {
      'KNN': KNeighborsClassifier,
      'SVC': SVC,
      'DecisionTree': DecisionTreeClassifier,
      'RandomForestClassifier': RandomForestClassifier,
      'LogisticRegression': LogisticRegression,
      'MLP': MLPClassifier,
      'AdaBoost': AdaBoostClassifier,
      'Bagging': BaggingClassifier
    }
    self.clf_call = classifers[clf_strategy]
    self.fit_data = fit_data
    self.fit_labels = fit_labels

  def tune(self, params_ranges, tst_data, tst_labels, max_only=True):
    max_score = 0.0
    max_params = None
    all_params = {}
    keys = list(params_ranges.keys())
    for values in product(*params_ranges.values()):
      # classification & score with params
      params = {keys[i]:value for i, value in enumerate(values)}
      clf = self.clf_call(**params).fit(self.fit_data, self.fit_labels)
      score = clf.score(tst_data, tst_labels)
      
      # book keeping
      all_params[values] = score
      if max_score < score:
        max_score = score
        max_params = params

    # setting class best params and current score
    self.max_params = max_params
    self.score = max_score
    if max_only:
      return max_params
    else:
      return all_params

  def score(self, params, tst_data, tst_labels):
    clf = self.clf_call(**params).fit(self.fit_data, self.fit_labels)
    return clf.score(tst_data, tst_labels)



def string_from_dict(dictionary):
  s = ""
  i = 0
  for key, val in dictionary.items():
    if i > 0:
      s += ","
    s += str(key)+"="+str(val)
    i += 1
  return s