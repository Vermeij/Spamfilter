import pickle
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
import nltk
from HTMLParser import HTMLParser
import re

from reademails import *
# Create a stripper for parsing text files formatted in HTML 
class MLStripper(HTMLParser):
	def __init__(self):
		self.reset()
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)

# Instantiate the HTMLparser and fed it HTML
def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()

# Manually remove additional HTML tags and other substrings which are not preferred to be features
def clean_html(text):
	# Remove inline JavaScript/CSS
	cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", text)
	# Remove html comments 
	cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
	# Remove remaining HTML tags:
	cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
	# Remove whitespace
	cleaned = re.sub(r"&nbsp;", " ", cleaned)
	# Remove URLS
	cleaned = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', cleaned)
	# Remove other characters
	cleaned = re.sub(r"\n", "", cleaned)
	cleaned = re.sub(r"^$", "", cleaned)
	cleaned = re.sub("''|,", "", cleaned)
	cleaned = re.sub(r"\n", " ", cleaned)
	cleaned = re.sub(r'\s\s+', ' ', cleaned)
	cleaned = re.sub(r'-', ' ', cleaned)
	cleaned = re.sub(r'_', ' ', cleaned)
	cleaned = re.sub(r'=',' ', cleaned)
	cleaned = re.sub(r'/',' ', cleaned)
	cleaned = re.sub(r"utf",'', cleaned)
	cleaned = re.sub(r"  ", " ", cleaned)
	return cleaned

# Clean, tokenize, lemmatize en uncapitalize email content.
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


#Initialize features. 
#When setting is 'bow', features are the amount of words in each email. 
#When setting is not 'bow', features are the presence of words in an email.

def get_features(text, setting):
	# Frequent and non-informative words in emails (e.g. 'the') are filtered.
	stoplist = stopwords.words('english')
	if setting=='bow':
		return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
	else:
		return {word: True for word in preprocess(text) if not word in stoplist}

# Remove features that appear only in one email 
def removeuniques(featureslist):
	countwords = {}
	uniquewords = []
	# For each word, count the amount of mails in which it is present.
	for features in all_features:
		for word in features[0].keys():
			if word in countwords:
				countwords[word] += 1
			else:
				countwords[word] = 1
	# For each word that appears in only one mail, remove the feature
	uniquewords = [k for k, v in countwords.iteritems() if v == 1]
	for features in all_features:
		for word in uniquewords:
			if word in features[0]:
				del(features[0][word])

# Returns list with all features with their labels, ready to be trained.
def buildfeaturelist(LOADFEATURES, SPAMFOLDERS, HAMFOLDERS):
	# Only generate new features when LOADFEATURES is False, otherwise load saves features.
	if LOADFEATURES is True:
		all_features = pickle.load(open('all_features.p', 'rb'))
	if LOADFEATURES is False:
		spam = []
		ham = []
		
		# Initialize emaillist
		for folder in SPAMFOLDERS:
			spam += init_emaillist(folder)
		for folder in HAMFOLDERS:
			ham += init_emaillist(folder)

		# Label all the emails
		all_emails = [(email, 'spam') for email in spam]
		all_emails += [(email, 'ham') for email in ham]
		all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

		# Remove all features that appear in only one mail
		removeuniques(all_features)

		# Save features for future computational efficieny	
		pickle.dump(all_features, open('all_features.p', 'wb'))	
	return all_features


