from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.metrics import classification_report as cr
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
from itertools import product


def encode(vector):
	le = preprocessing.LabelEncoder()
	le.fit(vector)
	vector = le.transform(vector)
	return vector

def encodeAll(mat):
	categoricalIndices = [1,2,3]
	for i in categoricalIndices:
		mat[:,i] = encode(mat[:,i])
	return mat.astype(np.float)

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors, tl):
	true = 0
	total = 0
	testlabels = []
	z = []
	misclassified = 0
	for feature in featureVectors:
		z = z + predict(model, feature).astype(np.int).tolist()
	testlabels = tl.tolist()
	# print cr(testlabels, z)
	return z

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
	def __init__(self, clfs, voting='hard', weights=None):

		self.clfs = clfs
		self.named_clfs = {key:value for key,value in _name_estimators(clfs)}
		self.voting = voting
		self.weights = weights

	def fit(self, X, y):
		if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
			raise NotImplementedError('Multilabel and multi-output'\
									  ' classification is not supported.')

		if self.voting not in ('soft', 'hard'):
			raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
							 % voting)

		if self.weights and len(self.weights) != len(self.clfs):
			raise ValueError('Number of classifiers and weights must be equal'
							 '; got %d weights, %d clfs'
							 % (len(self.weights), len(self.clfs)))

		self.le_ = LabelEncoder()
		self.le_.fit(y)
		self.classes_ = self.le_.classes_
		self.clfs_ = []
		for clf in self.clfs:
			fitted_clf = clone(clf).fit(X, self.le_.transform(y))
			self.clfs_.append(fitted_clf)
		return self

	def predict(self, X):
		if self.voting == 'soft':

			maj = np.argmax(self.predict_proba(X), axis=1)

		else:  # 'hard' voting
			predictions = self._predict(X)

			maj = np.apply_along_axis(
									  lambda x:
									  np.argmax(np.bincount(x,
												weights=self.weights)),
									  axis=1,
									  arr=predictions)

		maj = self.le_.inverse_transform(maj)
		return maj

	def predict_proba(self, X):
		avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
		return avg

	def transform(self, X):
		if self.voting == 'soft':
			return self._predict_probas(X)
		else:
			return self._predict(X)

	def get_params(self, deep=True):
		if not deep:
			return super(EnsembleClassifier, self).get_params(deep=False)
		else:
			out = self.named_clfs.copy()
			for name, step in six.iteritems(self.named_clfs):
				for key, value in six.iteritems(step.get_params(deep=True)):
					out['%s__%s' % (name, key)] = value
			return out

	def _predict(self, X):
		return np.asarray([clf.predict(X) for clf in self.clfs_]).T

	def _predict_probas(self, X):
		return np.asarray([clf.predict_proba(X) for clf in self.clfs_])


file = open("../newdataset/training_set.csv")
featureVectors = []
testVectors = []
testlabels = []

N = 250000
M = 250000
for line in file:	
	vector = line.strip().lower().split(',')
	featureVectors.append(vector)

file = open("../newdataset/testing_set.csv")
for line in file:	
	vector = line.strip().lower().split(',')
	testVectors.append(vector)

file = open("../newdataset/testing_labels.csv")
for line in file:   
	vector = line.strip().lower()
	testlabels.append(vector)

train = np.array(featureVectors)
test = np.array(testVectors)

testlabels = np.array(testlabels)
y = train[:, -1]
train = train[:, :-1]

mat = np.zeros((np.size(train, 0) + np.size(test, 0), np.size(test, 1)))
mat = mat.astype('str')
mat[:np.size(train, 0), :] = train
mat[np.size(train, 0):, :] = test
# featureVectors.extend(testVectors)
mat = encodeAll(mat)
X = mat[:N, :]
# y = mat[:N, -1]

# test = encodeAll(test)
test = mat[N:N+M, :]
print test.shape
print X.shape
print y.shape

mylabels = np.zeros(np.size(testlabels, 0) + np.size(y, 0))
mylabels = mylabels.astype('str')
mylabels[:np.size(testlabels)] = testlabels
mylabels[np.size(testlabels):] = y

mylabels = encode(mylabels)

y = mylabels[np.size(testlabels):]
tl = mylabels[:np.size(testlabels)]
# clf1 = DecisionTreeClassifier()
# clf2 = RandomForestClassifier()
# clf3 = GaussianNB()
# eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')
print "Training started"
eclf = RandomForestClassifier()
eclf.fit(X, y)
print "Training done"
z = classify(eclf, test, tl)

target_names = ['type2', 'type1', 'type3', 'type4', 'type5']
N = np.size(z, 0)
f = open('./myprediction.txt', 'w')
for i in xrange(N):
	f.write(target_names[z[i]] + '\n')















