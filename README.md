# NeuralNetworkImplementation
Handwritten Character Recognition using Neural Network implemented from scratch without any library function. The python code demonstrates the raw implementation of Neural Network.
A neural network from scratch is created herein with 1 input layer, 1 hidden layer, and 1 output layer. The number of layers are selected for ease of understanding and the same Neural Network can be easily scaled to comprise multiple hidden layers for further improvement in the accuracy.
The inputs are images converted into CSV format comprising 785 columns. First column holds the image class label which is used to train and test the designed neural network. the remaining 784 columns hold the pixel value data of an handwritten character image of size 28x28 pixels (i.e., 784 in total)
The training and testing data ratio can be provided as a hyperparameter by the user along with the learning rate (alpha) and number of iterations (epoches).
