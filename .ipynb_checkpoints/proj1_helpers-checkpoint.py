# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def standardise(tx):
    """
    Standardises over N samples each of the D features in the provided dataset
    
    tx: dataset (NxD)
    returns: standardised dataset
    """
    mean = np.mean(tx, axis=0)
    tx = tx - mean
    std = np.std(tx, axis=0)
    tx = tx / std
    return tx
    
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    
    x: dataset (N*D)
    y: predictions (N)
    returns: x_train, y_train, x_test, y_test
    """
    # set seed
    np.random.seed(seed)

    N = len(x)
    nb_train = int(N*ratio)
    nb_test = N-nb_train
    
    mask = np.random.rand(N) < ratio
    x_train = x[mask]
    y_train = y[mask]
    x_test = x[~mask]
    y_test = y[~mask]
    
    return x_train, y_train, x_test, y_test

def compute_accuracy(weights, labels, data):
    """
    Computes the accuracy on a test set for the given weights vector
    
    weights: weight vector to test
    labels: target test labels
    data: test samples
    returns: accuracy measure (TruePos+TrueNeg)/Total
    """
    labels_pred = predict_labels(weights, data)
    TN = ((labels == -1) & (labels_pred == -1)).sum()
    TP = ((labels == 1) & (labels_pred == 1)).sum()
    acc = (TN+TP)/len(labels)
    return acc


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
            
            
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    phi = np.zeros((x.shape[0], degree+1))
    curr_x = np.ones(x.shape)
    for i in range(degree+1):
        phi[:,i] = curr_x
        curr_x = curr_x*x
    return phi