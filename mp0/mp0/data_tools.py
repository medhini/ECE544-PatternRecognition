"""Processing data tools for mp0.
"""
from __future__ import division
import re
import numpy as np
import string

def title_cleanup(data):
	"""Remove all characters expect a-z, A-Z and spaces from the title,
	   then convert all characters to lower case.

	Args:
		data(dict): Key: article_id(int),
					Value: [title(str), positivity_score(float)](list)
	"""
	for key,value in data.items():
		
		allowed = string.ascii_lowercase[:]+string.ascii_uppercase[:]+' '

		for char in value[0]:
			if char not in allowed:
				value[0] = value[0].replace(char, "")

		data[key][0] = value[0].lower()
	pass


def most_frequent_words(data):
	"""Find the more frequeny words (including all ties), returned in a list.

	Args:
		data(dict): Key: article_id(int),
					Value: [title(str), positivity_score(float)](list)
	Returns:
		max_words(list): List of strings containing the most frequent words.
	"""
	max_words = []
	frequency = {}
	max_frequency = 1

	for key,value in data.items():
		wordList = value[0].split()
		for word in wordList:
			if word in frequency:
				frequency[word] = frequency[word] + 1
				if max_frequency < frequency[word]:
					max_frequency = frequency[word]
			else:
				frequency[word] = 1


	for key, value in frequency.items():
		if value == max_frequency:
			max_words.append(key)

	
	return max_words

def most_positive_titles(data):
	"""Computes the most positive titles.
	Args:
		data(dict): Key: article_id(int),
					Value: [title(str), positivity_score(float)](list)
	Returns:
		titles(list): List of strings containing the most positive titles,
					  include all ties.
	"""
	titles = []
	max_pos = next(iter(data.values()))[1]

	for key,value in data.items():
		if value[1]>max_pos:
			max_pos = value[1]

	for key,value in data.items():
		if value[1] == max_pos:
			titles.append(value[0])

	return titles


def most_negative_titles(data):
	"""Computes the most negative titles.
	Args:
		data(dict): Key: article_id(int),
					Value: [title(str), positivity_score(float)](list)
	 Returns:
		titles(list): List of strings containing the most negative titles,
					  include all ties.
	"""
	titles = []
	min_pos = next(iter(data.values()))[1]
	for key,value in data.items():
		if value[1]<min_pos:
			min_pos = value[1]

	for key,value in data.items():
		if value[1] == min_pos:
			titles.append(value[0])

	return titles


def compute_word_positivity(data):
	"""Computes average word positivity.
	Args:
		data(dict): Key: article_id(int),
					Value: [title(str), positivity_score(float)](list)
	Returns:
		word_dict(dict): Key: word(str), value: word_index(int)
		word_avg(numpy.ndarray): numpy array where element
								 #word_dict[word] is the
								 average word positivity for word.
	"""
	word_dict = {}
	word_avg = np.array([])
	counter = 0
	word_score = np.array([])
	word_count = np.array([])

	for key,value in data.items():
		words = value[0].split()
		for word in words:
			if word in word_dict:
				word_score[word_dict[word]] = word_score[word_dict[word]] + value[1]
				word_count[word_dict[word]] = word_count[word_dict[word]] + 1
			else:
				word_dict[word] = counter
				word_score = np.append(word_score,value[1])
				word_count = np.append(word_count, 1)
				counter = counter + 1

	for i in range(len(word_count)):
		word_avg = np.append(word_avg, word_score[i] / float(word_count[i]))

	return word_dict, word_avg


def most_postivie_words(word_dict, word_avg):
	"""Computes the most positive words.
	Args:
		word_dict(dict): output from compute_word_positivity.
		word_avg(numpy.ndarray): output from compute_word_positivity.
	Returns:
		words(list):
	"""
	words = []

	max_pos = word_avg[0]

	for key,value in word_dict.items():
		if max_pos < word_avg[value]:
			max_pos = word_avg[value]

	for key,value in word_dict.items():
		if(word_avg[value] == max_pos):
			words.append(key)

	return words


def most_negative_words(word_dict, word_avg):
	"""Computes the most negative words.
	Args:
		word_dict(dict): output from compute_word_positivity.
		word_avg(numpy.ndarray): output from compute_word_positivity.
	Returns:
		words(list):
	"""
	words = []

	min_pos = word_avg[0]

	for key,value in word_dict.items():
		if min_pos > word_avg[value]:
			min_pos = word_avg[value]

	for key,value in word_dict.items():
		if(word_avg[value] == min_pos):
			words.append(key)
		
	return words

