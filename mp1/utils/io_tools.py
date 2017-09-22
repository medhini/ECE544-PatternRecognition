"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io
from skimage import filters
from os import listdir
import csv

def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """

    f = open(data_txt_file, 'r')
    filenames = os.listdir(image_data_path)

    N = len(filenames)
    images = np.zeros((N,28,28))
    labels = np.zeros((N))

    data = {}
    i = 0

    for line in f:
        words = line.split()
        words[1] = int(words[1])
        filename = os.path.join(image_data_path, words[0])
        images[i] = io.imread(filename)
        labels[i]= words[1]
        i = i+1

    data['image'] = images
    data['label'] = labels
    
    pass
    return data


def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).
    """
    images = data['image']
    labels = data['label']
    f = open(data_txt_file, 'wb')
    
    filewriter = csv.writer(f, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['id','prediction'])

    for i in range(0,len(labels)):
        filename = 'test' + str(i).zfill(5) + '.png'
        filewriter.writerow([filename,labels[i]])

    pass
