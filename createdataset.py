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


file = open("kdd.data")
featureVectors = []
for line in file:	
	vector = line.strip().lower().split(',')
	if vector[-1] == 'normal.':
		vector[-1] = 'Normal'
	elif vector[-1] in ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.']:
		vector[-1] = 'R2L'
	elif vector[-1] in ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']:
		vector[-1] = 'U2R'
	elif vector[-1] in ['ipsweep.', 'nmap.', 'portsweep.', 'satan.']:
		vector[-1] = 'Probing'
	elif vector[-1] in ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.']:
		vector[-1] = 'DOS'
	featureVectors.append(vector)	

random.seed(170)
random.shuffle(featureVectors)

featureVectors = np.array(featureVectors);

labels = featureVectors[:, -1]
featureVectors = featureVectors[:, :-1]

num_probes = 100
sz = ((np.size(featureVectors, 0), num_probes))
probes = np.random.uniform(low=0.0, high=50, size=sz)
probes = probes.astype('str')
b = np.zeros((np.size(featureVectors,0), np.size(featureVectors, 1) + num_probes + 1))
b = b.astype('str')
b[:,:np.size(featureVectors,1)] = featureVectors
b[:, np.size(featureVectors, 1):-1] = probes
b[:, -1] = labels
featureVectors = b.tolist()

print "Feature Matrix Done"


N = 250000
f = open('./newdataset/training_set.csv', 'w')
M = np.size(featureVectors, 1)
for i in xrange(N):
	for j in xrange(M - 1):
		f.write(featureVectors[i][j] + ',')
	f.write(featureVectors[i][j + 1] + '\n')

print "Training Data Done"

N = 244021
# print len(featureVectors)
f = open('./newdataset/testing_set.csv', 'w')
f1 = open('./newdataset/testing_labels.csv', 'w')
M = len(featureVectors[0])
for i in xrange(N):
	for j in xrange(M - 2):
		f.write(featureVectors[250000 + i][j] + ',')
	f.write(featureVectors[250000 + i][j + 1] + '\n')
	f1.write(featureVectors[250000 + i][j + 2] + '\n')

# N = 244021
# print len(featureVectors)
# f = open('testing_set.csv', 'w')
# M = len(featureVectors[0])
# for i in xrange(N):
# 	for j in xrange(M - 1):
# 		f.write(featureVectors[250000 + i][j] + ',')
# 	f.write(featureVectors[250000 + i][j + 1] + '\n')

print "Testing Data Done"









