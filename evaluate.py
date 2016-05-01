import collections
from nltk import classify
import nltk

# Get performance measures for classifier
def evaluate(train_set, test_set, classifier, name):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	for i, (features, label) in enumerate(test_set):
		refsets[label].add(i)
		observed = classifier.classify(features)
		testsets[observed].add(i)
	# Get accuracy on training set, test set and get positive and negative recall.
	trainacc = 100 * classify.accuracy(classifier, train_set)
	testacc = 100 * classify.accuracy(classifier, test_set)
	spam_false = 100 - nltk.recall(refsets['spam'], testsets['spam'])*100
	ham_false = 100 - nltk.recall(refsets['ham'], testsets['ham'])*100
	return trainacc, testacc, spam_false, ham_false

# Print average performance measures from n iterations
def printperformance(name, perfmeasures_n, n):
	perfmeasures_avg = [x / n for x in perfmeasures_n]
	print('\n--------------------------\n'+name+'\n--------------------------')
	print('Accuracy on the training set = ' + str(round(perfmeasures_avg[0],2)))
	print('Accuracy on the test set = ' + str(round(perfmeasures_avg[1],2)))
	print('Percentage of spam not filtered:' + str(round(perfmeasures_avg[2], 2)))
	print('Percentage of ham filtered as spam:' + str(round(perfmeasures_avg[3],2)))
