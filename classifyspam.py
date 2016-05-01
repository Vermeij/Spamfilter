import random
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from operator import add

from evaluate import *
# Split all features in a train set and a test set
def buildsets(features, SAMPLE_PROPORTION):
	train_size = int(len(features) * SAMPLE_PROPORTION)
	train_set = features[:train_size]
	test_set = features[train_size:]
	return train_set, test_set


# Build classifiers for given features and print results of nfold tests.
def buildclassifiers(featureslist, SAMPLE_PROPORTION, n):
	classnames = ['Naive Bayes', 'Logistic Regression', 'Linear SCV']
	allclassifiers = []
	for name in classnames:
		for i in range(n):
			random.shuffle(featureslist)
			train_set, test_set = buildsets(featureslist, SAMPLE_PROPORTION)

			if name == 'Naive Bayes':
				spamclassifier = NaiveBayesClassifier.train(train_set)
			if name == 'Logistic Regression':
				spamclassifier = SklearnClassifier(LogisticRegression())
				spamclassifier.train(train_set)
			if name == 'Linear SCV':
				spamclassifier = SklearnClassifier(LinearSVC())
				spamclassifier.train(train_set)
			perfmeasures_i = evaluate(train_set, test_set, spamclassifier, name)
			if i == 0:
				perfmeasures_n = perfmeasures_i
			else:
				perfmeasures_n = map(add, perfmeasures_n, perfmeasures_i)
	
		# Store last classifier built per model
		allclassifiers.append(spamclassifier)
		
		# Print performance measures per classifier
		printperformance(name, perfmeasures_n, n)	
		
	return allclassifiers
