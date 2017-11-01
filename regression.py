import numpy as np
import scipy.io


def sigmoid(z):
    val = 1/(1+np.exp(-1*z))
    return val


def mle(trn_y, trn_x, weight):
    individual_mle = np.zeros(trn_y.shape)
    for i in range(trn_y.size):
        individual_mle[i, 0] += (-1*(trn_y[i, 1]*np.dot(trn_x[i, :].T, weight) -
                                     np.log(1 + np.exp(np.dot(weight.T, trn_x[i, :])))))
    total_mle = 0
    for i in range(individual_mle.size):
        total_mle += individual_mle[i, 0]
    return total_mle


def gradient(trn_y, trn_x, weight):
    for i in range(trn_y.size):
        prediction = -1*np.dot(weight, trn_x[i, :].T)
        print(prediction)
        sig_prediction = sigmoid(prediction)
        print(sig_prediction)
        grad = -1*(trn_y[i, 0]*trn_x[i, :]) + (1 - sig_prediction)*trn_x[i, :]
    combined_gradient = np.zeros(1, grad.shape(2))
    for i in range(grad.shape(1)):
        combined_gradient += grad[i, :]
    return combined_gradient


def gradient_descent(trn_y, trn_x, learning_rate):
    weight = np.zeros((1, trn_x.shape[1]))
    for i in range(1000):
        weight += learning_rate*gradient(trn_y, trn_x, weight)
        if i % 50 == 0:
            print('mle', mle(trn_y, trn_x, weight))
    return weight


data = scipy.io.loadmat('data1.mat')
x_trn = np.asmatrix(data['X_trn'])
y_trn = np.asmatrix(data['Y_trn'])
weights = gradient_descent(y_trn, x_trn, 0.0001)
print(x_trn, y_trn)



