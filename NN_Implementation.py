import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

start_time = time.time()

Classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
           7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
           13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
           20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R',
           27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y',
           34: 'Z', 35: '@', 36: '#', 37: '$', 38: '&'}

data = pd.read_csv('/Users/Abhi/Desktop/CSV Files Input/Updated_converted_images_1000.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_train = data[0:30000].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0

data_dev = data[30000:m].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

'''
def init_params():
    W1 = np.random.rand(39, 784)*0.05
    b1 = np.random.rand(39, 1)*0.05
    W2 = np.random.rand(39, 39)*0.05
    b2 = np.random.rand(39, 1)*0.05
    return W1, b1, W2, b2
'''
def init_params():
    W1 = np.random.normal(size=(39, 784)) * np.sqrt(1. / (784))
    b1 = np.random.normal(size=(39, 1)) * np.sqrt(1. / 39)
    W2 = np.random.normal(size=(39, 39)) * np.sqrt(1. / 78)
    b2 = np.random.normal(size=(39, 1)) * np.sqrt(1. / (784))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z -= np.max(Z, axis=0)  # Max value is subtracted
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy percentage: ", 100*get_accuracy(predictions, Y))

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 501)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    Label = Y_train[index]
    print("Prediction | Label: ", Classes[prediction[0]]," | ", Classes[Label])

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

testing_values = 5
for x in range(testing_values):
    test_prediction(x, W1, b1, W2, b2)

end_time = time.time()
total_time = end_time - start_time
print("Total Running time is: ", total_time, " seconds")