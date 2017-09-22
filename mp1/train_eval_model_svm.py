"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import sklearn
from models.support_vector_machine import SupportVectorMachine
import random
from sklearn.utils import shuffle

def train_model(data, model, learning_rate=0.001, batch_size=100,
                num_steps=100, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.

    batch_epoch_num = data['label'].shape[0] // batch_size
    epochs = 1

    for i in range(epochs):
        if shuffle:
            data['image'],data['label'] = sklearn.utils.shuffle(data['image'],data['label'], random_state=0)
        print(i)
        for j in range(0,data['label'].shape[0],batch_size):
            image_batch = data['image'][j:(j+batch_size)]
            label_batch = data['label'][j:(j+batch_size)]
            print(j)
            for k in range(num_steps):
                update_step(image_batch, label_batch, model, learning_rate)
    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(image_batch)
    gradient = model.backward(f,label_batch)
    model.w = model.w - learning_rate*gradient


def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(data['image'])
    loss = model.loss(f,data['label'])

    y_predict = model.predict(f)

    count = 0
    for i in range(len(data['label'])):
        if data['label'][i] == y_predict[i]:
            count = count + 1

    acc = (count/len(data['label']))*100

    return loss, acc
