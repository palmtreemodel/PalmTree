import os
import sys

#getting all file list in a directory
def get_files(directory):
	file_list = []
	for root, dirc, files in os.walk(directory):
		for file in files:
			file_list.append(os.path.join(root, file))


	return file_list