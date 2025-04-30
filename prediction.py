from scipy.optimize import curve_fit
import numpy as np
import torch
import math


def func(x, a, b, c):
    return a*x**2 + b*x + c


def naive_prediction(x, y, d, k, lr, dr, epoch):
    # Prediction distance
    d_0 = d * pow(dr, (epoch + 1) / (x+1))
    d = d_0 + x

    x_arr = torch.arange(1, x+1)
    m, n = y.shape
    y_t = y.T
    y_new = torch.zeros(n)
    for i in range(n):
        para, _ = curve_fit(func, x_arr.numpy(), y_t[i].numpy())
        a, b, c = para
        y_new[i] = func(d, a, b, c)
    return y_new


def adaptive_prediction(x, y, d, k, lr, dr, epoch):
    x_arr = torch.arange(1, x+1)
    m, n = y.shape
    y_t = y.T
    y_new = torch.zeros(n)
    for i in range(n):
        para, _ = curve_fit(func, x_arr.numpy(), y_t[i].numpy())
        a, b, c = para

        # Prediction distance
        m = abs(2*a*x + b)
        dm = abs(2*a)
        d_0 = k * (m/(dm * lr + 0.000001))
        d = (1 - math.exp(-d_0))*12 + x

        y_new[i] = func(d, a, b, c)
    return y_new


def tf_prediction(weights_dict, sample_epochs, predict_epoch):

    # stores predicted weights
    pred_weights = []

    # prediction sample size
    x = [i+1 for i in range(sample_epochs-1)]

    # iterates through each layer of the model
    for i in range(len(weights_dict[0])):
        layer_pred = []
        layer_weights = []
        layer_shape = np.shape(weights_dict[0][i])

        # flattens and stores weights of layer i
        for j in range(1, sample_epochs):
            layer_weights.append(np.array(weights_dict[j][i]).flatten())

        # iterates through all weights of layer i and stores predicted weights
        for j in range(len(layer_weights[0])):
            y = []
            for k in range(sample_epochs - 1):
                y.append(layer_weights[k][j])
            para, _ = curve_fit(func, x, y)
            a, b, c = para
            layer_pred.append(func(predict_epoch, a, b, c))

        pred_weights.append(np.reshape(layer_pred, layer_shape))

    print("\n***Weight Prediction***")
    return pred_weights
