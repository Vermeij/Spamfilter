import nltk
import os
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import collections
from nltk import NaiveBayesClassifier, classify
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from operator import add
import pickle

#import HTMLstripper
from striphtml import *

stoplist = stopwords.words('english')
SAMPLE_PROPORTION = 0.8

HAMFOLDERS = ['Data/beck-s/']
SPAMFOLDERS = ['Data/BG/2004/']
LOADFEATURES = False

def read_files(path):
	for root, dir_names, file_names in os.walk(path):
		for path in dir_names:
			read_files(os.path.join(root, path))
		for file_name in file_names:
			file_path = os.path.join(root, file_name)
			if os.path.isfile(file_path):
				past_header, lines = False, []
				f = open(file_path)
				for line in f:
					if past_header:
						lines.append(line)
					elif line == '\n':
						past_header = True
				content = '\n'.join(lines)
				f.close()
				yield file_path, content

def init_emaillist(path):
	email_list = []
	for file_path, content in read_files(path):
		#strip stuff and HTML removed to preprocess
		email_list.append(content)
	return email_list

def preprocess(sentence):
	word_list = []
	try:
		sentence = strip_tags(sentence)
		sentence = clean_html(sentence) 
		sentence =	word_tokenize(sentence.decode('utf-8','ignore').encode('utf-8'))
	except UnicodeDecodeError:
		pass
	for word in sentence:
		try:
			word_list += [WordNetLemmatizer().lemmatize(word.lower())]
		except UnicodeDecodeError:
			pass
	return word_list

def get_features(text, setting):
	if setting=='bow':
		return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
	else:
		return {word: True for word in preprocess(text) if not word in stoplist}

def buildsets(features, SAMPLE_PROPORTION):
	train_size = int(len(features) * SAMPLE_PROPORTION)
	train_set = features[:train_size]
	test_set = features[train_size:]
	return train_set, test_set

def evaluate(train_set, test_set, classifier, name):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (features, label) in enumerate(test_set):
		refsets[label].add(i)
		observed = classifier.classify(features)
		testsets[observed].add(i)
	trainacc = 100 * classify.accuracy(classifier, train_set)
	testacc = 100 * classify.accuracy(classifier, test_set)
	spam_false = 100 - nltk.recall(refsets['spam'], testsets['spam'])*100
	ham_false = 100 - nltk.recall(refsets['ham'], testsets['ham'])*100
	#print('\n-----------------------\n'+name+'\n-----------------------')
	#print('Accuracy on the training set = ' + str(round(100*classify.accuracy(classifier, train_set),2)))
	#print('Accuracy on the test set = ' + str(round(100*classify.accuracy(classifier, test_set),2)))
	#print 'pos precision (share of true spam over all called spam)', nltk.precision(refsets['spam'], testsets['spam'])
	#print('Percentage of spam not filtered:' + str(round(100-nltk.recall(refsets['spam'], testsets['spam'])*100, 2)))
	#print 'neg precision (share of true ham over all called ham):', nltk.precision(refsets['ham'], testsets['ham'])
	#print('Percentage of ham filtered as spam:' + str(round(100- nltk.recall(refsets['ham'], testsets['ham'])*100,2)))
	return trainacc, testacc, spam_false, ham_false

def removeuniques(featureslist):
	countwords = {}
	uniquewords = []
	for mail in all_features:
		for word in mail[0].keys():
			if word in countwords:
				countwords[word] += 1
			else:
				countwords[word] = 1
	uniquewords = [k for k, v in countwords.iteritems() if v == 1]
	for features in all_features:
		for word in uniquewords:
			if word in features[0]:
				del(features[0][word])

if LOADFEATURES is True:
	all_features = pickle.load(open('all_features.p', 'rb'))
if LOADFEATURES is False:
	spam = []
	ham = []
	for folder in SPAMFOLDERS:
		spam += init_emaillist(folder)
	for folder in HAMFOLDERS:
		ham += init_emaillist(folder)
	all_emails = [(email, 'spam') for email in spam]
	all_emails += [(email, 'ham') for email in ham]


	all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]
	#print(all_emails[0:3])
	#print(all_features[0:3])
	removeuniques(all_features)
	pickle.dump(all_features, open('all_features.p', 'wb'))	


print('__________________________\nTotal size = ' + str(len(all_features)) + ' emails')
print('Training set size = ' + str(int(len(all_features)*SAMPLE_PROPORTION)) + ' emails')
print('Test set size = ' + str(len(all_features) - int(len(all_features)*SAMPLE_PROPORTION)) + ' emails')

n = 3

def buildclassifiers(featureslist, SAMPLE_PROPORTION, n):
	#classnames = ['Logistic Regression']
	classnames = ['Naive Bayes', 'Logistic Regression', 'Linear SCV']
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
			perfmeasuresi = evaluate(train_set, test_set, spamclassifier, name)
			if i == 0:
				perfmeasuresn = perfmeasuresi
			else:
				perfmeasuresn = map(add, perfmeasuresn, perfmeasuresi)
	
		perfmeasuresavg = [x / n for x in perfmeasuresn]
		print('\n-----------------------\n'+name+'\n-----------------------')
		print('Accuracy on the training set = ' + str(round(perfmeasuresavg[0],2)))
		print('Accuracy on the test set = ' + str(round(perfmeasuresavg[1],2)))
		print('Percentage of spam not filtered:' + str(round(perfmeasuresavg[2], 2)))
		print('Percentage of ham filtered as spam:' + str(round(perfmeasuresavg[3],2)))


buildclassifiers(all_features, SAMPLE_PROPORTION, n)
# Performance imporovement
	# Overfitting
	# Look for Bigrams and Trigrams

# Explain why models perform this way
# Save classifiers
