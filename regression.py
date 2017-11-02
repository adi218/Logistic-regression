import numpy as np
import scipy.io
import math


def sigmoid(z):
    val = 1/(1+math.exp(-1*z))
    return val


def mle(trn_y, trn_x, weight):
    individual_mle = np.zeros(trn_y.shape)
    for i in range(trn_y.size):
        individual_mle[i, 0] = -1*(trn_y[i, 0]*np.dot(trn_x[i, :], weight.T) -
                                     np.log(sigmoid(-np.dot(weight, trn_x[i, :].T))))
    total_mle = 0
    for i in range(individual_mle.size):
        total_mle += individual_mle[i, 0]
    return total_mle


def gradient(trn_y, trn_x, weight):
    grad = np.zeros(trn_x.shape)
    for i in range(trn_y.size):
        prediction = -1*np.dot(weight, trn_x[i, :].T)
        # print(prediction[0, 0])
        sig_prediction = sigmoid(prediction[0, 0])
        grad[i, :] = (trn_y[i, 0] - sig_prediction) * trn_x[i, :]
    combined_gradient = np.zeros((1, grad.shape[1]))
    for i in range(grad.shape[0]):
        combined_gradient += grad[i, :]
    return combined_gradient


def gradient_descent(trn_y, trn_x, learning_rate):
    weight = np.zeros((1, trn_x.shape[1]))
    for i in range(1000):
        weight += learning_rate*gradient(trn_y, trn_x, weight)
        # if i % 50 == 0:
        #     print('mle', mle(trn_y, trn_x, weight))
    return weight


data = scipy.io.loadmat('data2.mat')
x_trn = np.asmatrix(data['X_trn'])
y_trn = np.asmatrix(data['Y_trn'])
weights = gradient_descent(y_trn, x_trn, 0.0001)
prediction = np.matmul(x_trn, weights.T)
prob = np.zeros(y_trn.shape)
for i in range(prediction.size):
    prob[i, 0] = sigmoid(prediction[i, 0])
    print(prob[i, 0], y_trn[i, 0])
print(weights)

# print(x_trn, y_trn)



