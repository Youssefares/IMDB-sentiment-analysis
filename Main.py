from DataSetReader import DataSetReader
from PreProcess import PreProcess
from Vectorizer import Vectorizer
from Classify import Classify
from NewClassifier import Classifier
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def clean_string(str):
  newstr = ''
  noise = ['[',']','{','}',"'",'(',')',',']
  for c in str:
    if c == ' ':
      newstr+='\n'
    elif c not in noise:
      newstr+= c
  return newstr

def get_plot_arrs(params,acc_num, items):
  plt_arr = [acc[0] for acc in items]
  plt_arr = [plt_arr[i] for i in range (0,acc_num)]
  ticks_arr = [clean_string(acc[1].__str__())+clean_string(acc[2].__str__())+'\n Parameters: '+
               clean_string(params.__str__())for acc in v]
  ticks_arr = [ticks_arr[i] for i in range (0,acc_num)]
  return plt_arr,ticks_arr

def plot_clfs(width,max_num,acc_sep,param_sep):
  for key, value in clf_dict.items():
    print(key)
    plt_acc = []
    plt_tks = []
    my_dpi = 192
    fig = plt.figure(figsize=(4096 / my_dpi, 2160 / my_dpi), dpi=my_dpi)
    acc_sep = acc_sep
    param_sep = param_sep
    acc_sep += len(value)/20
    param_sep += len(value)/10
    for k, v in value.items():
      plt.title(key.__str__())
      plt_arr, plt_arr_ticks = get_plot_arrs(k,max_num, v)
      print(plt_arr, plt_arr_ticks)
      if not plt_acc:
        plt_acc = plt_arr
        plt_tks = plt_arr_ticks
      else:
        plt_acc += plt_arr
        plt_tks += plt_arr_ticks
    ind = np.array(1)
    x = 1
    for i in range(1, len(plt_acc)):
      if ((i) % max_num == 0):
        x += param_sep * width
      else:
        x += acc_sep * width
      ind = np.append(ind, x)
    ind = ind[0:len(plt_acc)]
    colors =['#009688','#35a79c','#54b2a9','#65c3ba','#83d0c9','#8fd4ce','#9bd9d3']
    colors =[colors[i] for i in range(0,max_num)]
    plt.bar(ind, plt_acc, width=width,color=colors)
    plt.xticks(ind, plt_tks)
  plt.show()


dsr = DataSetReader(directory="../aclImdb/")

tr_data = dsr.labelled_string_data('train')
# Data split by label
tr_negative, tr_positive = tr_data[:len(tr_data)//2], tr_data[len(tr_data)//2:]

# small set of data of size 1000
tr_small = tr_negative[:10]+tr_positive[:10]


tst_data = dsr.labelled_string_data('test')

tst_negative, tst_positive = tst_data[:len(tst_data)//2], tst_data[len(tst_data)//2:]

tst_small = tst_negative[:10]+tst_positive[:10]


# Preprocessing combinations
cleaning_operations = ['remove_stopwords','lemmatize','stemmingLS','stemmingPS','stemmingSB'];
cleaning_operations = ['remove_stopwords','lemmatize'];

combinations = [i for j in range(len(cleaning_operations)) for i in itertools.combinations(cleaning_operations,j+1)]

# Vectorizers
vec_list = [['tfidf',{}],['count',{}],['wordembedd',{'min_count':1}],['fasttext',{}]]

# Classifiers
clf_list = [['KNN',{}],['SVC',{}],['DecisionTree',{}],['RandomForestClassifier',{}],['LogisticRegression',{}]
  ,['MLP',{}],['AdaBoost',{}],['Bagging',{}]]

clf_list = [['RandomForestClassifier',{'n_estimators': [i for i in range(10,80,10)]}],
            ['KNN',{'n_neighbors':[i for i in range(1,6,2)]}]
           ]
clf_list=[['RandomForestClassifier',{'n_estimators':[80]}]]
vec_list = [['fastext',{'min_count':1}]]


# ex: {'KNN': [(acc, vectorization, preprocessing_ops)], 'LR': ... }
clf_dict = {clf[0]: {} for clf in clf_list}
for vec in vec_list:
  print()
  print('Vectorization: '+vec[0].__str__())
  for combination in combinations:
    print('Preprocessing: ' + combination.__str__())
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
          clf_dict[cl[0]][key] += [(value, vec,combination)]
        else:
          clf_dict[cl[0]][key] = [(value, vec, combination)]

for key,value in clf_dict.items():
  for k,v in value.items():
    v = (sorted(v, key=lambda x: x[0], reverse=True))
    clf_dict[key][k] = v

print(clf_dict)

plot_clfs(0.15,3,1.2,2.8)
# width = 0.15
# max_num = 3
# acc_sep = 1.2
# param_sep = 2.8
