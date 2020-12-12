from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import nn
""" Reading saved Data  """

X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')


def plot_history(history):
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    n = 4000
    plt.plot(range(history['epochs'])[:n],
             history['train_loss'][:n], label='train_loss')
    plt.plot(range(history['epochs'])[:n],
             history['test_loss'][:n], label='test_loss')
    plt.title('train & test loss')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(history['epochs'])[:n],
             history['train_acc'][:n], label='train_acc')
    plt.plot(range(history['epochs'])[:n],
             history['test_acc'][:n], label='test_acc')
    plt.title('train & test accuracy')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()


X = X_train
y = y_desired

neural_net = nn.NeuralNetwork([50, 4, 3, 4], seed=0)

history = neural_net.train(X=X_train, y=y_desired, X_t=X_test, y_t=y_test_desired, batch_size=1, epochs=100, learning_rate=0.4, print_every=1, validation_split=0.2,
                           plot_every=10)
