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

            
def standardise(tx, mean=None, std=None):
    """
    Standardises over N samples each of the D features in the provided dataset
    If a mean and std are provided, then standardisation is done with these values
    
    tx: dataset (NxD)
    mean: mean to substract
    std: std to divide by
    returns: standardised dataset, mean and std
    """
    if((mean is None) and (std is None)):
        mean = np.mean(tx, axis=0)
        std = np.std(tx, axis=0)
    tx = tx - mean
    tx = tx / std
    return tx, mean, std

def normalise(tx):
    """
    Normalises over N samples each of the D features in the provided dataset
    
    tx: dataset (NxD)
    returns: normalised dataset
    """
    mean = np.mean(tx, axis=0)
    tx = tx - mean
    std = np.std(tx, axis=0)
    tx = tx / std
    
    min_ = np.min(tx, axis=0)
    max_ = np.max(tx, axis=0)
    tx = (tx-min_)/(max_-min_)
    return tx


def mass_abs(tx) :
    """
    Manage the -999s in the DER_mass_MMC column (to do it we found an interval in which the distribution of (-1, 1) is pretty similar as the one of         -999, the interval is (60, 80). The masses are going to be uniformely distributed over this interval), substract 125 (Approximate of the mass of the     Higgs boson) and compute the absolute value of it.
    
    tx: The dataset in which we have the DER_mass_MMC column we want to modifiy
    """
    x = tx.copy()
    nb999 = np.sum(x[:,0] == -999)
    uni = np.random.uniform(60, 80, nb999)
    for i in range(x.shape[1]) :
        if (x[i, 0] == -999) :
            x[i, 0] = uni[int(np.random.randint(nb999, size = 1))]
    x[:,0] = np.abs(x[:,0] - 125)
    return x
    
    
def build_poly(tx, degree) :
    """
    Polynomial extension from j=1 to degree of each components of tx 
    
    returns: augmented matrix
    """
    shape = tx.shape
    poly = np.zeros((shape[0], shape[1] * degree))
    poly[:,:shape[1]] = tx
    for deg in range(2, degree + 1) :
        for j in range(0, shape[1]) :
            poly[:, shape[1] * (deg - 1) + j] = tx[:,j] ** deg
    return poly


def add_offset(tx):
    """
    Adds an offset column, whcih is composed only of 1s.
    
    returns: augmented matrix
    """
    return np.c_[np.ones(len(tx[:,0])), tx]
            
    
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)    
    random_pick = np.random.default_rng().choice(x.shape[0], int(x.shape[0]*ratio), replace=False)
    
    x_train = x[random_pick]
    y_train = y[random_pick]
    x_test = np.delete(x, random_pick)
    y_test = np.delete(y, random_pick)
    
    return x_train, y_train, x_test, y_test

def sigmoid(t):
    """apply the sigmoid function on t."""
    """TODO plus joli/mieux"""
    ret = 1 / (1 + np.exp(-t))
    return np.clip(ret, 10**(-10), 1 - 10**(-10))


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

def compute_accuracy_log_reg(weights, labels, data):
    """
    Computes the accuracy on a test set for the given weights vector
    
    weights: weight vector to test
    labels: target test labels
    data: test samples
    returns: accuracy measure (TruePos+TrueNeg)/Total
    """
    labels_pred = predict_labels(weights, data)
    TN = ((labels == 0) & (labels_pred == -1)).sum()
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
            