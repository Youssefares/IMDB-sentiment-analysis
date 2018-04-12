from sklearn.ensemble import RandomForestClassifier


class Classify:
  """
  Class for classifying data after vectorization
  """
  def __init__(self, tr_data,tst_data):
    self.tr_data = tr_data
    self.tst_data = tst_data
    self.tr_vecs = [d[1] for d in self.tr_data]
    self.tr_labels = [d[2] for d in self.tr_data]
    self.tst_vecs = [d[1] for d in self.tst_data]
    self.tst_labels = [d[2] for d in self.tst_data]

  def knn(self):
    from sklearn.neighbors import KNeighborsClassifier
    scores = []
    scores.append([0,0])
    knn = KNeighborsClassifier(n_neighbors=1, weights='distance').fit(self.tr_vecs, self.tr_labels)
    h_score = knn.score(self.tst_vecs,self.tst_labels)
    scores.append([1,h_score])
    i=0
    while scores[i+1][1] >= scores[i][1]:
      i+=1
      k=i+1
      knn = KNeighborsClassifier(n_neighbors=k, weights='distance').fit(self.tr_vecs, self.tr_labels)
      h_score = knn.score(self.tst_vecs,self.tst_labels)
      scores.append([k,h_score])
      print(scores)
    return scores

  def decision_trees(self):
    from sklearn.tree import DecisionTreeClassifier
    results = []
    clf = DecisionTreeClassifier().fit(self.tr_vecs,self.tr_labels)
    score = clf.score(self.tst_vecs,self.tst_labels)
    results.append(['Gini',score])

    clf = DecisionTreeClassifier(criterion='entropy').fit(self.tr_vecs,self.tr_labels)
    score = clf.score(self.tst_vecs,self.tst_labels)
    results.append(['Entropy', score])
    return results

  def random_forests(self, criterion='gini'):
    from sklearn.ensemble import RandomForestClassifier
    i=0
    n_trees = 100
    clf = RandomForestClassifier(criterion=criterion,n_estimators=n_trees).fit(self.tr_vecs,self.tr_labels)
    h_score = clf.score(self.tst_vecs,self.tst_labels)
    scores = []
    scores.append([0,0])
    scores.append([n_trees, h_score])
    while scores[i + 1][1] >= scores[i][1]:
      n_trees+=100
      clf = RandomForestClassifier(criterion=criterion, n_estimators=n_trees).fit(self.tr_vecs,self.tr_labels)
      h_score = clf.score(self.tst_vecs,self.tst_labels)
      scores.append([n_trees,h_score])
      i+=1
    scores.remove(scores[-1])
    n_trees=scores[-1][0]
    h_score=scores[-1][1]
    return scores, n_trees,h_score

  def bagging(self):
    from sklearn.ensemble import BaggingClassifier
    cart = RandomForestClassifier()
    num_trees = 100
    clf = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, oob_score=True)
    clf.fit(self.tr_vecs, self.tr_labels)
    scores = clf.score(self.tst_vecs, self.tst_labels)
    return scores