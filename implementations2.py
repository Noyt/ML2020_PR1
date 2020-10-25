import numpy as np

def reg_logistic_regression_norm(y, tx, lambda_, initial_w, max_iters, gamma, norm = 'l2'):
    """
    Regularized logistic regression using gradient descent
    
    y: targets
    tx: training data samples
    lambda_: regularization parameter
    initial_w: initial weight vector
    max_iters: number of steps to run for the GD
    gamma: learning rate
    norm: regularisation norm (l1 or l2)
    """

    def calculate_penalized_loss(y, tx, w, lambda_, norm):
        """compute the loss: negative log likelihood + regularization."""
        penalty = 0
        if (norm == 'l1') :
            penalty = (lambda_/2)*np.linalg.norm(w, ord = 1)
        elif (norm == 'l2') :
            penalty = (lambda_/2)*np.linalg.norm(w, ord = 2) 
        return compute_loss_classification(y, tx, w) + penalty

    def calculate_penalized_gradient(y, tx, w, lambda_, norm):
        """compute the gradient of penalized loss."""
        sig = sigmoid(tx.dot(w))
        pen_grad = 0
        if (norm == 'l1') :
            pen_grad = (w > 0) * 2 - 1
        elif (norm == 'l2') :
            pen_grad = w
        return tx.T.dot(sig-y)/tx.shape[0] + lambda_ * pen_grad
    
    threshold = 1e-8
    previous_loss = None

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_penalized_loss(y, tx, w, lambda_, norm)
        grad = calculate_penalized_gradient(y, tx, w, lambda_, norm)
        w = w - gamma*grad
        
        # converge criterion
        if iter > 0 and np.abs(loss - previous_loss) < threshold:
            break
        else:
            previous_loss = loss
    
    return w, loss