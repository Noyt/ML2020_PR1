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

def add_off(tx):
    """
    Adds an offset column, whcih is composed only of 1s.
    
    returns: augmented matrix
    """
    return np.c_[np.ones(len(tx[:,0])), tx]

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
            
#####################################################################################################################
            
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/output.csv'

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

degree = 9
lambda_ = 1.38e-10

np.random.seed(1)

tx = mass_abs(tX)
tx = np.delete(tx, [15, 18, 20], axis = 1)
tx, mean_train, std_train = standardise(tx)
poly_tx = build_poly(tx, degree = degree)
poly_tx = add_off(poly_tx)

tx_test = mass_abs(tX_test)
tx_test = np.delete(tx_test, [15, 18, 20], axis = 1)
tx_test, _, _ = standardise(tx_test, mean_train, std_train)
poly_tx_test = build_poly(tx_test, degree = degree)
poly_tx_test = add_off(poly_tx_test)

w, _ = ridge_regression(y, poly_tx, lambda_)

y_pred = predict_labels(w, poly_tx_test)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
