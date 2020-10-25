# -*- coding: utf-8 -*-

"""ML methods"""

import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    
    y: targets
    tx: training data samples
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    return: optimal weights and mse
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
    max_iters: number of steps to run for the SGD
    gamma: learning rate
    return: optimal weights and mse
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
    Least squares regression using normal equations
    
    y: targets
    tx: training data samples
    return: optimal weights and mse
    """
    # Compute closed form solution
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    
    #Compute mse loss
    err = y-tx.dot(w)
    N = y.shape[0]
    loss = (1/(2*N))*((err.T).dot(err))
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    y: targets
    tx: training data samples
    lambda_: regularisation parameter
    return: optimal weights and mse
    """
    N = y.shape[0]
    I = np.identity(tx.shape[1])
    lb = lambda_*(2*N)
    w = np.linalg.solve(tx.T.dot(tx)+lb*I, tx.T.dot(y))
    
    #Compute mse loss
    err = y-tx.dot(w)
    loss = (1/(2*N))*((err.T).dot(err))
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
        penalty = (lambda_/2)*np.linalg.norm(w, ord = 2) 
        return compute_loss_classification(y, tx, w) + penalty

    def calculate_penalized_gradient(y, tx, w, lambda_):
        """compute the gradient of penalized loss."""
        sig = sigmoid(tx.dot(w))
        return tx.T.dot(sig-y) + lambda_ * w
    
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


################ Helpers ###################

def compute_loss_regression(y, tx, w, loss='MSE'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if loss not in ['MSE', 'MAE']:
        raise Exception("Loss function {} is not supported".format(loss))
    else:
        err = y-tx.dot(w)
        N = y.shape[0]
        if loss=='MSE':
            return (1/(2*N))*((err.T).dot(err))
        elif loss=='MAE':
            return (1/(2*N))*(np.abs(err).sum())
        
        
def compute_loss_classification(y, tx, w):
        """compute the loss: negative log likelihood."""
        sig = sigmoid(tx.dot(w))
        return - (y * np.log(sig) + (1-y) * np.log(1 - sig)).sum()

    
def sigmoid(t):
    """apply the sigmoid function on t."""
    ret = 1 / (1 + np.exp(-t))
    return np.clip(ret, 10**(-10), 1 - 10**(-10)) # Escape overflows
    
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]