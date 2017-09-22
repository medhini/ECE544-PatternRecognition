
import os
import numpy as np 
import skimage
from skimage import io
import csv

f = open('./data/train_lab.txt', 'r')
filenames = os.listdir('./data/image_data')

N = len(filenames)
images = np.zeros((N,28,28))
labels = np.zeros((N))

data = {}
i = 0

for line in f:
	words = line.split()
	words[1] = int(words[1])
	filename = os.path.join('./data/image_data', words[0])
	images[i] = io.imread(filename)
	labels[i]= words[1]
	i = i+1


data['image'] = images
data['label'] = labels

print data
# f = open('./data/write.csv', 'wb')
# images = data['image']
# labels = data['label']

# filewriter = csv.writer(f, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)

# filewriter.writerow(['id','prediction'])
# for i in range(0,len(labels)):
# 	filename = 'test_' + str(i).zfill(5) + '.png'
# 	filewriter.writerow([filename,labels[i]])