from __future__ import division
import numpy as np

data = {}

file = open("./data/fake.txt", "r")
for line in file:
    parts = line.split('\t')
    parts[2] = parts[2][:-1] #Removing the last character '\n'
    parts[2] = float(parts[2])
    parts[0] = int(parts[0])
    values = [parts[1], parts[2]]
    data[parts[0]] = values

print data[000][0]
#print data
# word_dict = {}
# word_avg = np.array([])
# counter = 0
# word_score = np.array([])
# word_count = np.array([])

# for key,value in data.items():
#     words = value[0].split()
#     for word in words:
#         if word in word_dict:
#             word_score[word_dict[word]] = word_score[word_dict[word]] + value[1]
#             word_count[word_dict[word]] = word_count[word_dict[word]] + 1
#         else:
#             word_dict[word] = counter
#             word_score = np.append(word_score,value[1])
#             word_count = np.append(word_count, 1)
#             counter = counter + 1

# for i in range(len(word_count)):
#     word_avg = np.append(word_avg, word_score[i] / float(word_count[i]))

# words = []

# min_pos = word_avg[0]

# for key,value in word_dict.items():
#     if min_pos > word_avg[value]:
#         min_pos = word_avg[value]

# for key,value in word_dict.items():
#     if(word_avg[value] == min_pos):
#         words.append(key)

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


for key, value in word_dict.items():
    print key, word_avg[value]

words = []

max_pos = word_avg[0]

for key,value in word_dict.items():
    if max_pos < word_avg[value]:
        max_pos = word_avg[value]

for key,value in word_dict.items():
    if(word_avg[value] == max_pos):
        words.append(key)

print words
print max_pos

mwords = []

min_pos = word_avg[0]

for key,value in word_dict.items():
    if min_pos > word_avg[value]:
        min_pos = word_avg[value]

for key,value in word_dict.items():
    if(word_avg[value] == min_pos):
        mwords.append(key)

print mwords
print min_pos