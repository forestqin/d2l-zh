import torch
import random
import numpy as np
from torch.nn import Sequential
import os
import sys
from utils import load_data_fashion_mnist, train_ch3

os.chdir(sys.path[0])
# print(torch.cuda.is_available())

batch_size = 256
num_inputs = 784
num_outputs = 10


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b) 

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])


def updater(batch_size):
    def sgd(params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    return sgd([W, b], lr, batch_size)

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1

def main():

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def main1():
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    net = Sequential(torch.nn.Flatten(), torch.nn.Linear(num_inputs, num_outputs))
    net.apply(init_weights)

    loss = torch.nn.CrossEntropyLoss(reduction="none")

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)




if __name__=="__main__":
    # main()
    main1()
