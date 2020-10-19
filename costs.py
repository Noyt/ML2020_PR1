# -*- coding: utf-8 -*-
"""Costs toolbox"""

def compute_loss(y, tx, w, loss='MSE'):
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