import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt


def sigmoid(z):
    val = 1/(1+np.exp(-1*z))
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


def gradient(trn_y, trn_x, weight, lamb):
    grad = np.zeros(trn_x.shape)
    for i in range(trn_y.size):
        prediction = np.dot(weight, trn_x[i, :].T)
        # print(prediction)
        sig_prediction = sigmoid(prediction[0, 0])
        grad[i, :] = (trn_y[i, 0] - sig_prediction) * trn_x[i, :]
    combined_gradient = np.zeros((1, grad.shape[1]))
    for i in range(grad.shape[0]):
        combined_gradient += grad[i, :]
    combined_gradient = combined_gradient - lamb * combined_gradient
    return combined_gradient


def gradient_descent(trn_y, trn_x, learning_rate, selected_lamb):
    weight = np.zeros((1, trn_x.shape[1]))
    for i in range(5000):
        weight += learning_rate*gradient(trn_y, trn_x, weight, selected_lamb)
        if i % 500 == 0:
            print(i, 'mle', mle(trn_y, trn_x, weight))
            print(weight)
    return weight


def regularize(x, y, k_f, learning_rate_f):
    lambdas = []
    gen = 0
    losses = []
    total_loss = []
    while gen < 0.03:
        lambdas.append(gen)
        gen = gen + 0.01
    for i in range(len(lambdas)):
        probable = lambdas[i]
        for l in range(k_f):
            idx = []
            for m in range(int(x.shape[0] / k_f)):
                idx.append(int(l*x.shape[0] / k_f + m))
            # print(idx)
            tst_x = x[idx, :]
            tst_y = y[idx, :]
            trn_idx = []
            for j in range(x.shape[0]):
                if j not in idx:
                    trn_idx.append(j)
            trn_x = x[trn_idx, :]
            trn_y = y[trn_idx, :]

            theta_temp = gradient_descent(trn_y, trn_x, learning_rate_f, probable)
            loss_temp = mle(tst_y, tst_x, theta_temp)
            losses.append(loss_temp)
        temp_loss = 0
        for a in losses:
            temp_loss += a
        total_loss.append(temp_loss/k_f)
        losses = []
        print(probable, total_loss[i])
    print(lambdas[total_loss.index(min(total_loss))], (min(total_loss)))
    return lambdas[total_loss.index(min(total_loss))]

data = scipy.io.loadmat('data2.mat')
x_trn = np.asmatrix(data['X_trn'])
intercept = np.ones((x_trn.shape[0], 1))
x_trn = np.append(x_trn, intercept, 1)
# print(x_trn)
y_trn = np.asmatrix(data['Y_trn'])
x_tst = np.asmatrix(data['X_tst'])
y_tst = np.asmatrix(data['Y_tst'])
intercept = np.ones((x_tst.shape[0], 1))
x_tst = np.append(x_tst, intercept, 1)

learning_rate = 0.01

lam = regularize(x_trn, y_trn, 2, learning_rate)
weights = gradient_descent(y_trn, x_trn, learning_rate, lam)
prediction_tst = np.matmul(x_tst, weights.T)
prob_tst = np.zeros(y_tst.shape)
# # print(prediction)
for i in range(prediction_tst.size):
    prob_tst[i, 0] = sigmoid(prediction_tst[i, 0])
    # if prob_tst[i, 0] <= 0.5:
    #     prob_tst[i, 0] = 0
    # else:
    #     prob_tst[i, 0] = 1
#     # print(prob[i, 0], y_tst[i, 0])
# print(weights)
#
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter([x_tst[:, 0]], [y_tst[:, 0]], c='blue', label='given')
ax1.scatter([x_tst[:, 0]], [prob_tst[:, 0]], c='red', label='predicted')
plt.legend(loc='upper left')
plt.show()

prediction_trn = np.matmul(x_trn, weights.T)
prob_trn = np.zeros(y_trn.shape)
# print(prediction)
for i in range(prediction_trn.size):
    prob_trn[i, 0] = sigmoid(prediction_trn[i, 0])
    # print(prob[i, 0], y_tst[i, 0])
# print(weights)

x2 = np.zeros((x_trn.shape[0], 1))
for i in range(y_trn.size):
    x2[i, 0] = -1*(weights[0, 0]*x_trn[i, 0])/weights[0, 1]
for i in range(y_trn.size):
    if y_trn[i, 0] == 0:
        plt.scatter([x_trn[i, 0]], [x_trn[i, 1]], c='red')
    else:
        plt.scatter([x_trn[i, 0]], [x_trn[i, 1]], c='blue')
plt.plot(x_trn[:, 0], x2[:, 0], color='black')
plt.ylim([-1, 5])
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter([x_trn[:, 0]], [y_trn[:, 0]], c='blue', label='given')
ax1.scatter([x_trn[:, 0]], [prob_trn[:, 0]], c='red', label='predicted')
plt.legend(loc='upper left')
plt.show()
