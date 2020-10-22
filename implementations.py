# -*- coding: utf-8 -*-

"""ML methods"""

import numpy as np
from costs import *
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    y: targets
    tx: training data samples
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    """

    def compute_gradient(y, tx, w):
        """Computes the gradient for least squares"""
        # ***************************************************
        # compute gradient and error vector
        # ***************************************************
        e = y-tx.dot(w)
        N = y.shape[0]
        return (-1/N)*(tx.T).dot(e)


    def gradient_descent(y, tx, initial_w, max_iters, gamma):
        """Gradient descent algorithm."""
        # Define parameters to store w and loss
        w = initial_w
        for n_iter in range(max_iters):
            # ***************************************************
            # compute gradient and loss
            # ***************************************************
            loss = compute_loss_regression(y, tx, w, 'MSE')
            grad = compute_gradient(y, tx, w)
            # ***************************************************
            # update w by gradient
            # ***************************************************
            w = w - gamma*grad

        return w, loss
    
    return gradient_descent(y, tx, initial_w, max_iters, gamma)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    
    y: targets
    tx: training data samples
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    """
    
    def compute_stoch_gradient(y, tx, w, batch_size):
        """
        Compute a stochastic gradient from just 
        few examples n and their corresponding y_n labels.
        """
        minibatch_y, minibatch_tx = next(batch_iter(y, tx, batch_size))
        e = minibatch_y-minibatch_tx.dot(w)
        N = minibatch_y.shape[0]
        return (-1/N)*(minibatch_tx.T).dot(e)


    def stochastic_gradient_descent(
            y, tx, initial_w, batch_size, max_iters, gamma):
        """Stochastic gradient descent algorithm."""
        w = initial_w
        for n_iter in range(max_iters):
            loss = compute_loss_regression(y, tx, w, loss='MSE')
            grad = compute_stoch_gradient(y, tx, w, batch_size)
            w = w - gamma*grad
        return w, loss
    
    return stochastic_gradient_descent(y, tx, initial_w, batch_size=1, max_iters=max_iters, gamma=gamma)
    
    
def least_squares(y, tx):
    """
    Calculate the least squares solution.
    return: optimal weights and rmse
    """
    # Compute closed form solution
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    
    #Compute mse loss
    err = y-tx.dot(w)
    N = y.shape[0]
    loss = (1/(2*N))*((err.T).dot(err))
    #TODO
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    lambda_: regularization parameter
    returns: loss, w
    """
    I = np.identity(tx.shape[1])
    lb = lambda_*(2*len(y))
    w = np.linalg.solve(tx.T.dot(tx)+lb*I, tx.T.dot(y))
    
    #Compute rmse loss
    err = y-tx.dot(w)
    N = y.shape[0]
    loss = (1/(2*N))*((err.T).dot(err))
    # TODO now is mse, for comparison sake
    #loss = np.sqrt(2*loss)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    
    y: targets
    tx: training data samples
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    """
    
    def calculate_gradient(y, tx, w):
        """compute the gradient of loss."""
        sig = sigmoid(tx.dot(w))
        return tx.T.dot(sig - y)
    
    threshold = 1e-8
    previous_loss = None

    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss_classification(y, tx, w)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad
        # converge criterion
        if iter > 0 and np.abs(loss - previous_loss) < threshold:
            break
        else:
            previous_loss = loss
            
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    
    y: targets
    tx: training data samples
    lambda_: regularization parameter
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    """

    def calculate_penalized_loss(y, tx, w, lambda_):
        """compute the loss: negative log likelihood + regularization."""
        sig = sigmoid(tx.dot(w))
        trd = (lambda_/2)*np.linalg.norm(w)**2
        return - (y * np.log(sig) + (1-y) * np.log(1 - sig)).sum() + trd

    def calculate_penalized_gradient(y, tx, w, lambda_):
        """compute the gradient of penalized loss."""
        sig = sigmoid(tx.dot(w))
        return tx.T.dot(sig-y)+lambda_*w
    
    threshold = 1e-8
    previous_loss = None

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_penalized_loss(y, tx, w, lambda_)
        grad = calculate_penalized_gradient(y, tx, w, lambda_)
        w = w - gamma*grad
        
        # converge criterion
        if iter > 0 and np.abs(loss - previous_loss) < threshold:
            break
        else:
            previous_loss = loss
    
    return w, loss
