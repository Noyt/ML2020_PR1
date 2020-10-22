# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def cross_validation_visualization_loss(lambds, mse_tr, mse_te, param_name, ax):
    """visualization the curves of mse_tr and mse_te."""
    ax.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    ax.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    ax.set_xlabel(param_name)
    ax.set_ylabel("rmse")
    ax.set_title("cross validation")
    ax.legend(loc=2)
    ax.grid(True)
    
def cross_validation_visualization_metric(lambds, metric_tr, metric_te, param_name, metric_name, ax):
    """visualization the curves of the metric for train and test."""
    ax.semilogx(lambds, metric_tr, marker=".", color='b', label='train {}'.format(metric_name))
    ax.semilogx(lambds, metric_te, marker=".", color='r', label='test {}'.format(metric_name))
    ax.set_xlabel(param_name)
    ax.set_ylabel("{}".format(metric_name))
    ax.set_title("cross validation")
    ax.legend(loc=2)
    ax.grid(True)
