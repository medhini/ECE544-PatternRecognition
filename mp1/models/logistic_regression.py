"""
Implements logistic regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LogisticRegression(LinearModel):
    """
    """

    def sigmoid(val):
        return 1/(1+np.exp(-1*val))

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """

        # grad_1 = sigmoid(f);
        # gradient = np.transpose(self.x) * (grad_1 - y)

        gradient = np.mean((-1*np.transpose(self.x)*y*np.exp(-1*y*f))/(1+ np.exp(-1*y*f)), axis=1)
        return gradient

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
        f(numpy.ndarray): Output of forward operation, dimension (N,).
        y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
        (float): average log loss.
        """

        l = np.mean(np.log(1+np.exp(-1*y*f)))
        # grad_1 = sigmoid(f);
        # N = len(y)
        # l = (-np.transpose(y) * np.log(grad_1) - np.transpose(1-y) * np.log(1-grad_1))/N;

        return l

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """

        y_predict = np.zeros(len(f))

        for i in range(len(f)):
            if f[i]>=0.5:
                y_predict[i] = 1
            else:
                y_predict[i] = -1

        return None
