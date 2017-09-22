"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    images = data['image']

    if process_method == 'default':
        
        images = images/255
        
        for i in range(len(images)):
            images[i] = filters.laplace(images[i], ksize=11, mask=None)
        
        data['image'] = images
        data = remove_data_mean(data)
        N = len(images)
        images2 = np.zeros((N, 28*28))

        for i in range(len(images)):
            images2[i]= images[i].flatten()

        data['image'] = images2
        #print(data['image'])
        pass
        
    elif process_method == 'raw':
        images = images/255
        data['image'] = images
        data = remove_data_mean(data)
        N = len(images)
        images2 = np.zeros((N, 28*28))

        for i in range(len(images)):
            images2[i]= images[i].flatten()

        data['image'] = images2

        #print(data['image'])
        pass
    elif process_method == 'custom':
        pass

    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    image_mean = np.ndarray((28,28))
    images = data['image']

    for i in range(len(images)):
        images_mean = image_mean + images[i]

    image_mean = image_mean/len(images)
    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    image_mean = compute_image_mean(data)
    images = data['image']
    images = images - image_mean
    data['image'] = images

    return data
