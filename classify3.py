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

stoplist = stopwords.words('english')
SAMPLE_PROPORTION = 0.8
SKIP_FILES = {'cmds'}

SOURCES = [
	('Data/beck-s', 'ham')
#	('Data/BG', 'spam')
]

def read_files(path):
	for root, dir_names, file_names in os.walk(path):
		for path in dir_names:
			read_files(os.path.join(root, path))
		for file_name in file_names:
			if file_name not in SKIP_FILES:
				file_path = os.path.join(root, file_name)
				if os.path.isfile(file_path):
					past_header, lines = False, []
					f = open(file_path)
					for line in f:
						if past_header:
							lines.append(line)
						elif line == '\n':
							past_header = True
					f.close()
					content = '\n'.join(lines)
					yield file_path, content

def init_emaillist(path):
	email_list = []
	for path, classification in SOURCES:
		for file_path, content in read_files(path):
			email_list.append(content)
	return email_list

def classification():
	all_emails = []
	for path, classification in SOURCES:
		all_emails += [(content, classification) for content in init_emaillist(path)]
	return all_emails

all_emails = classification()
print all_emails[0:2]
print ('\nCorpus size = ' + str(len(all_emails)) + ' emails')
random.seed(1)
random.shuffle(all_emails)


def preprocess(sentence):
	# This must be removed to better place
	titleoption = True
	word_list = []
	if titleoption is True:
		sentencesplit = sentence.splitlines()
		for word in word_tokenize(sentencesplit[0].decode('utf-8', 'ignore').encode('utf-8')):
			word_list += ['TITLE' + WordNetLemmatizer().lemmatize(word.lower())]
	for word in word_tokenize(sentence.decode('utf-8', 'ignore').encode('utf-8')):
		word_list += [WordNetLemmatizer().lemmatize(word.lower())]
	return word_list


def get_features(text, setting):
	if setting=='bow':
		return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
	else:
		return {word: True for word in preprocess(text) if not word in stoplist}

def train(features, SAMPLE_PROPORTION):
	train_size = int(len(features) * SAMPLE_PROPORTION)
	train_set = features[:train_size]
	test_set = features[train_size:]
	print ('Training set size = ' + str(len(train_set)) + ' emails')
	print ('Test set size = ' + str(len(test_set)) + ' emails')
	return train_set, test_set

def evaluate(train_set, test_set, classifier, name):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (features, label) in enumerate(test_set):
		refsets[label].add(i)
		observed = classifier.classify(features)
		testsets[observed].add(i)
	print('\n ===='+name+'====')
	print('Accuracy on the training set = ' + str(round(100*classify.accuracy(classifier, train_set),2)))
	print('Accuracy on the test set = ' + str(round(100*classify.accuracy(classifier, test_set),2)))
	#print 'pos precision (share of true spam over all called spam)', nltk.precision(refsets['spam'], testsets['spam'])
	print('Percentage of spam not filtered:' + str(round(100-nltk.recall(refsets['spam'], testsets['spam'])*100, 2)))
	#print 'neg precision (share of true ham over all called ham):', nltk.precision(refsets['ham'], testsets['ham'])
	print('Percentage of ham filtered as spam:' + str(round(100- nltk.recall(refsets['ham'], testsets['ham'])*100,2)))

spam = []
ham = []
#spam = init_list('enron1/spam/')
#ham = init_list('enron1/ham/')

all_emails = [(email, 'spam') for email in spam]
all_emails += [(email, 'ham') for email in ham]
print ('\nCorpus size = ' + str(len(all_emails)) + ' emails')
#random.seed(1)
random.shuffle(all_emails)


nrevs = 0
for i in range(nrevs):
	random.shuffle(all_emails)
	
	all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]
	print(spam[2])
	print(preprocess(spam[2]))
	print(get_features(spam[2], 'bow'))
	train_set, test_set = train(all_features, SAMPLE_PROPORTION)

	NB_classifier = NaiveBayesClassifier.train(train_set)
	evaluate(train_set, test_set, NB_classifier, 'Naive Bayes')

	LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(train_set)
	evaluate(train_set, test_set, LogisticRegression_classifier, 'Logistic Regression')

	LinearSCV_classifier = SklearnClassifier(LinearSVC())
	LinearSCV_classifier.train(train_set)
	evaluate(train_set, test_set, LinearSCV_classifier, 'Linear SCV')


# Take Average of 10 evaluations
