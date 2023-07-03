import random
import torch
from torch.utils import data
from torch import nn
import numpy as np

print(torch.cuda.is_available())

def get_synthetic_data(nsamples, ture_w, ture_b):
    X = torch.normal(0, 1, (nsamples, len(ture_w)))
    y = torch.matmul(X, ture_w) + ture_b
    y += torch.normal(0, 0.01, y.shape)
    print(X[0], y[0])
    return X, y.reshape((-1,1))

def data_loader(features, labels, batch_size):
    nsamples = len(features)
    ids = list(range(nsamples))
    random.shuffle(ids)
    for i in range(0, nsamples, batch_size):
        sample_id = torch.tensor(ids[i:min(i+batch_size, nsamples)])
        yield features[sample_id], labels[sample_id]

def linear_reg(X, w, b):
    return torch.matmul(X, w) + b

def loss(pred_y, y):
    return (pred_y - y.reshape(pred_y.shape))**2/2

def Huber_loss(pred , true , sigma = 0.005):
    error = abs(pred.detach().numpy() - true.detach().numpy())
    return torch.tensor(np.where(error < sigma , error - sigma / 2 , 
                                 0.5 * sigma * error ** 2) , 
                                 requires_grad= True)
    # .mean()

def huber_loss(y_hat, y, beta = 0.005):
    error = torch.abs(y_hat - y.detach())
    return torch.where(error < beta , 0.5 * error ** 2 / beta, error - 0.5 * beta)

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def main():
    num_epochs = 3
    batch_size = 10
    lr = 0.02
    nsamples = 1000

    net = linear_reg

    ture_w = torch.tensor([2,-3.4])
    ture_b = torch.tensor(4.2)
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    features, labels = get_synthetic_data(nsamples, ture_w, ture_b)
    print("original process")
    for epoch in range(num_epochs):
        for X, y in data_loader(features, labels, batch_size):
            pred_y = net(X, w, b)
            l = loss(pred_y, y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
def data_loader_2(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def main2():
    print("torch process")
    num_epochs = 3
    batch_size = 10
    nsamples = 1000

    ture_w = torch.tensor([2,-3.4])
    ture_b = torch.tensor(4.2)

    features, labels = get_synthetic_data(nsamples, ture_w, ture_b)
    data_iter = data_loader_2((features, labels), batch_size)
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            pred_y = net(X)
            l = loss(pred_y, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


if __name__ == "__main__":
    # main()
    main2()