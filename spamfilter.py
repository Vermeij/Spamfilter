#import other procedures for this spamfilter
from reademails import *
from getfeatures import *
from evaluate import *
from classifyspam import *

# Specify the proportion of training set, rest is test set
SAMPLE_PROPORTION = 0.8

# Specify the folders which contain HAM and SPAM.
HAMFOLDERS = ['Data/beck-s/']
SPAMFOLDERS = ['Data/BG/2004/']

# Specify whether to load last created features. Set False when to create new features from emails. Set True when same features are used.
LOADFEATURES =  True

# When setting is 'bow', features are the amount of words in each email. 
# When setting is not 'bow', features are the presence of words in an email.
COUNTSETTING = 'notbow'

# Set number iterations to determine performance measures 
n = 10

# Get list of features with their labels spam or ham 
all_features = buildfeaturelist(LOADFEATURES, SPAMFOLDERS, HAMFOLDERS, COUNTSETTING)

# Print sizes of emails processed
print('==========SPAMFILTER==========\nTotal size = ' + str(len(all_features)) + ' emails')
print('Training set size = ' + str(int(len(all_features)*SAMPLE_PROPORTION)) + ' emails')
print('Test set size = ' + str(len(all_features) - int(len(all_features)*SAMPLE_PROPORTION)) + ' emails')

# Get classifiers and print performance measures from all features
all_classifiers = buildclassifiers(all_features, SAMPLE_PROPORTION, n)
