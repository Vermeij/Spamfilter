# Spamfilter
This is a spamfilter on the raw enron dataset.
The files are as follows:

### spamfilter.py
- Main script which calls the procedures of the other files
- Contains settings to change the spamfilter 

### reademails.py

- read_files(path):

⋅⋅⋅Walk through all files in given path and return content

- init_emaillsit(path):

⋅⋅⋅Initialize list with all raw email content

### getfeatures.py

- MLstripper(HTMLParser):

⋅⋅⋅Create a stripper class for parsing text files formatted in HTML

- strip_tags(html):

⋅⋅⋅Instantiate the HTMLparser and fed it HTML

- cleanhtml(text):

⋅⋅⋅Manually remove additional HTML tags and other substrings which are not preferred to be features.

- preprocess(sentence):

⋅⋅⋅Clean, tokenize, lemmatize en uncapitalize email content.

- get_features(text, setting):

⋅⋅⋅Initialize features. When setting is 'bow', features are the amount of words in each email. When setting is not 'bow', features are the presence of words in an email.

- removeuniques(featureslist):

⋅⋅⋅Remove features that appear only in one email

- buildfeaturelst(LOADFEATURES, SPAMFOLDERS, HAMFOLDERS):

⋅⋅⋅Returns list with all features with their labels, ready to be trained.
