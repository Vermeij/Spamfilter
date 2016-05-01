import os
 
# walk through all files in path and return path and content for each file.
def read_files(path):
	for root, dir_names, file_names in os.walk(path):
		for path in dir_names:
			read_files(os.path.join(root, path))
		for file_name in file_names:
			file_path = os.path.join(root, file_name)
			if os.path.isfile(file_path):
				# content in raw mails starts after whitespace, so only store content when past first header.
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

# Initiaize list with all raw email content
def init_emaillist(path):
	email_list = []
	for file_path, content in read_files(path):
		email_list.append(content)
	return email_list

