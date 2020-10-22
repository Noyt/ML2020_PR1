# -*- coding: utf-8 -*-
"""Costs toolbox"""
import numpy as np
from proj1_helpers import sigmoid

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
    print(sig)
    return - (y * np.log(sig) + (1-y) * np.log(1 - sig)).sum()


def compute_loss(learning_model, y, tx, w, loss='MSE'):
    """
    Matches to the appropirate loss
    """
    classification = ['logistic_regression', 'reg_logistic_regression']
    regression = ['least_squares', 'least_squares_gd', 'least_squares_sgd', 'ridge_regression']
    
    mod = learning_model.lower()
    
    if mod in classification:
        return compute_loss_classification(y, tx, w)
    elif mod in regression:
        return compute_loss_regression(y, tx, w, loss)
    else:
        raise Exception('Learning model {} is not supported'.format(learning_model))
    