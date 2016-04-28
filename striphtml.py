from HTMLParser import HTMLParser
import re


class MLStripper(HTMLParser):
	def __init__(self):
		self.reset()
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)

def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()

def clean_html(text):
	# First remove inline JavaScript/CSS:
	cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", text)
	# Then remove html comments. 
	cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
	# Next remove the remaining tags:
	cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
	# Finally deal with whitespace
	cleaned = re.sub(r"&nbsp;", " ", cleaned)
	#j	cleaned = re.sub(r"\n", "", cleaned)
	cleaned = re.sub(r"^$", "", cleaned)
	cleaned = re.sub("''|,", "", cleaned)
	cleaned = re.sub(r"  ", " ", cleaned)
	cleaned = re.sub(r"\n", " ", cleaned)
	cleaned = re.sub(r'\s\s+', ' ', cleaned)
	# remove URLS
	cleaned = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', cleaned)
	# remove -
	cleaned = re.sub(r'-', '', cleaned)
	cleand = re.sub(r'_', '', cleaned)
	return cleaned
