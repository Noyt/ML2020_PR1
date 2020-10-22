# -*- coding: utf-8 -*-
"""Cross validation toolbox"""

import matplotlib.pyplot as plt
import numpy as np
from costs import *
from implementations import *
from plots import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_single_fold(y, x, k_indices, k, metric, learning_model, **kwargs):
    """
    returns the loss (train and test) and the metric for the k'th set of indices
    
    y: targets
    x: data samples
    k_indices: array of indices used for k-fold validation
    k: index for the set of indices to use
    metric: (function) metric to compute (e.g. accuracy, recall)
    learning_model: str, method to employ (e.g. 'least_squares', 'ridge_regression')
    **kwargs: additional arguments for learning method (e.g. lambda, degree)
    returns: training loss, test loss, metric
    """
    # ***************************************************
    # get k'th subgroup in test, others in train
    # ***************************************************
    mask = np.array(range(len(k_indices))) == k
    test_indices = k_indices[mask].flatten()
    train_indices = k_indices[~mask].flatten()
    y_test = y[test_indices]
    x_test = x[test_indices]
    y_train = y[train_indices] 
    x_train = x[train_indices]
    # ***************************************************
    # learning
    # *************************************************** 
    w, loss_tr = run_model(learning_model, y_train, x_train, **kwargs) 
    loss_te = compute_loss(learning_model, y_test, x_test, w)
    # ***************************************************
    # metric
    # *************************************************** 
    metric_tr = metric(w, y_train, x_train)
    metric_te = metric(w, y_test, x_test)
    return loss_tr, loss_te, metric_tr, metric_te


def cross_validation(y, x, metric, learning_model, k_fold = 4, degree = 1, **kwargs):
    """
    returns the average losses (train and test) and the average measure over a k-fold
    
    y: targets
    x: data samples
    metric: (function) metric to compute (e.g. accuracy, recall)
    learning_model: str, method to employ (e.g. 'least_squares', 'ridge_regression')
    k_fold: number of folds to perform
    degree: degree of the polynomial expension
    **kwargs: additional arguments for learning method (e.g. lambda, degree)
    returns: mean training loss, test loss, training metric, test metric
    """
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    metrics_tr = []
    metrics_te = []
    
    # Polynomial expansion, normalisation and bias augmentation
    poly = build_poly(x, degree)
    
    # ***************************************************
    # cross validation
    # ***************************************************

    for k in range(len(k_indices)):
        loss_tr, loss_te, metric_tr, metric_te = cross_validation_single_fold(y, poly, k_indices, k, metric, learning_model, **kwargs)
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)
        metrics_tr.append(metric_tr)
        metrics_te.append(metric_te)
    return np.mean(losses_tr), np.mean(losses_te), np.mean(metrics_tr), np.mean(metrics_te)
        

def run_model(learning_model, y, x, **kwargs):
    """
    Run appropriate model with the corresponding parameters
    
    y: targets
    x: data samples
    learning_model: str, method to employ (e.g. 'least_squares_GD', 'ridge_regression')
    **kwargs: additional arguments for learning method (e.g. lambda, degree)
    returns: training loss, test loss, metric
    """
    mod = learning_model.lower()
    if mod == 'least_squares_gd':
        return least_squares_GD(y, x, kwargs['initial_w'], kwargs['max_iters'], kwargs['gamma'])
    elif mod == 'least_squares_sgd': 
        return least_squares_SGD(y, x, kwargs['initial_w'], kwargs['max_iters'], kwargs['gamma'])
    elif mod == 'least_squares':
        return least_squares(y, x)
    elif mod == 'ridge_regression':
        return ridge_regression(y, x, kwargs['lambda_'])
    elif mod == 'logistic_regression':
        return logistic_regression(y, x, kwargs['initial_w'], kwargs['max_iters'], kwargs['gamma'])
    elif mod == 'reg_logistic_regression':
        return reg_logistic_regression(y, x, kwargs['lambda_'], kwargs['initial_w'], kwargs['max_iters'], kwargs['gamma'], kwargs['norm'])
    else:
        raise Exception('Learning method {} is currently not supported.'.format(learning_model))
        


def cross_validation_hyper_search(y, x, param_name, search_space, metric, learning_model, k_fold, **kwargs):
    """
    TODO
    
    y: targets
    x: data samples
    param_name: str, parameter to search, must be the same as would be provided in kwargs
    search_space: iterable, values from which the parameter will be searched
    metric: (function) metric to compute (e.g. accuracy, recall)
    learning_model: str, method to employ (e.g. 'least_squares', 'ridge_regression')
    k_fold: number of folds to perform
    **kwargs: additional arguments for learning method (e.g. lambda, degree)
    returns: TODO
    """
    seed = 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    metric_tr = []
    metric_te = []
    
    # map to retreive best performing parameter TODO is selecting by the loss the best idea ?
    loss_to_param = {}
    
    # ***************************************************
    # cross validation
    # ***************************************************
    for param in search_space:
        kwargs[param_name] = param
        tmp_mse_tr = []
        tmp_mse_te = []
        tmp_metric_tr = []
        tmp_metric_te = []
        for k in range(len(k_indices)):
            loss_tr, loss_te, met_tr, met_te = cross_validation_single_fold(y, x, k_indices, k, metric, learning_model, **kwargs)
            tmp_mse_tr.append(loss_tr)
            tmp_mse_te.append(loss_te)
            tmp_metric_tr.append(met_tr)
            tmp_metric_te.append(met_te)
        # Appending mean measures over the folds
        mse_tr.append(np.mean(tmp_mse_tr))
        mse_te.append(np.mean(tmp_mse_te))
        metric_tr.append(np.mean(tmp_metric_tr))
        metric_te.append(np.mean(tmp_metric_te))
        
        loss_to_param[np.mean(tmp_mse_te)] = param
        
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10,7), sharex=True, sharey=False)
    cross_validation_visualization_loss(search_space, mse_tr, mse_te, param_name, axes[0])
    # TODO accuracy make generic
    cross_validation_visualization_metric(search_space, metric_tr, metric_te, param_name, 'accuracy', axes[1])
    
    return loss_to_param[np.min(list(loss_to_param.keys()))]