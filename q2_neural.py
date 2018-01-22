#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    # wrap around the data structure

    X = np.insert(X,H,[1.0],axis = 1) # M * (Dx + 1)
    # print 'data',data
    W1 = np.insert(W1,Dx,b1,axis = 0) # (Dx + 1) * H
    # print 'W1',W1
    ### YOUR CODE HERE: forward propagation
    #X W1 b1 -> Z1 f -> h w2 b2 -> Z2 CE -> a
    z1 = np.dot(X, W1) # M * H
    # print 'z1',z1
    # print 'z1',z1
    h = sigmoid(z1)  # M * H
    # print 'h',h
    # wrap around the data structure
    h = np.insert(h,H,[1.0],axis = 1) # M * (H + 1)
    # print 'h',h
    W2 = np.insert(W2, H, b2, axis = 0) # (H + 1) * Dy
    # print 'W2',W2
    # true second layer calculation
    z2 = np.dot(h,W2) # M* Dy
    # print 'z2',z2
    soft_max_z2 = softmax(z2) # M * Dy
    # print 'soft_max_z2',soft_max_z2
    matrix_cost = np.log(np.multiply(soft_max_z2, labels).sum(axis=1))
    cost = -1 * matrix_cost.sum()
    # print 'cost',cost
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    #   g_w1 = x_t * sigma3 & g_b1 = sigma3
    # | sigma 3(z1) = sigma2 * f'(z1)
    # | sigma 2(h) =  sigma1 * w2
    # | g_w2 = sigma1*h & g_b2 = sigma1
    # | sigma1 (Z2) = y_hat - y
    sigma_1 = np.subtract(soft_max_z2, labels) # M * Dy : ce/y_hat
    # print 'sigma_1', sigma_1
    gradW2 = np.dot(np.transpose(h),sigma_1) # (H+1) * Dy : ce/w2
    # print 'gradW2 before ', gradW2
    # extract
    gradb2 = gradW2[H,:] # 1 * H
    # print 'gradb2', gradb2
    temp_gradW2 = gradW2
    gradW2 = gradW2[0:H,:] # H * Dy
    # print 'gradW2 after ', gradW2
    #
    sigma_2 = np.dot(sigma_1, np.transpose(W2)) [:,0:H]# M * H + 1  ce/y_hat  ce/h
    # print 'sigma_2',sigma_2
    sigma_3 = np.multiply(sigmoid_grad(h[:,0:H]), sigma_2)# M * H
    # print 'sigma_3',sigma_3
    gradW1 = np.dot(np.transpose(X), sigma_3) # (Dx + 1)  * H
    # print 'gradW1 ', gradW1
    gradb1 = gradW1[Dx,:] # 1 * H
    # print 'gradb1 ', gradb1
    gradW1 = gradW1[0:Dx,:]

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
