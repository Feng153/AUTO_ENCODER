import torch
import numpy as np
import pandas as pd
from rbm import RBM
from data_charts.load_dataset import MNIST
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 2500),
        torch.nn.Sigmoid(),
        torch.nn.Linear(2500, 10),
        torch.nn.Softmax(dim=1),
    )
    model.cuda(rbm.device)
    return model


def generate_batches(x, y, batch_size=64):
    x = x[:int(x.shape[0] - x.shape[0] % batch_size)]
    x = torch.reshape(x, (x.shape[0] // batch_size, batch_size, x.shape[1]))
    y = y[:int(y.shape[0] - y.shape[0] % batch_size)]
    y = torch.reshape(y, (y.shape[0] // batch_size, batch_size))
    return {'x': x, 'y': y}


def test(model, train_x, train_y, test_x, test_y, epoch):
    criterion = torch.nn.CrossEntropyLoss()

    # put data into gpu
    test_x1 = test_x.to(rbm.device)
    test_y1 = test_y.to(rbm.device)
    train_y1 = train_y.to(rbm.device)
    train_x1 = train_x.to(rbm.device)

    output_test = model(test_x1)
    loss_test = criterion(output_test, test_y1).item()
    output_test = torch.argmax(output_test, axis=1)
    acc_test = torch.sum(output_test == test_y1).item() / test_y1.shape[0]

    output_train = model(train_x1)
    loss_train = criterion(output_train, train_y1).item()
    output_train = torch.argmax(output_train, axis=1)
    acc_train = torch.sum(output_train == train_y1).item() / train_y1.shape[0]

    return epoch, loss_test, loss_train, acc_test, acc_train


def train(model, x, y, train_x, train_y, test_x, test_y, epochs=5):
    dataset = generate_batches(x, y)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training = trange(epochs)
    progress = []
    for epoch in training:
        running_loss = 0
        acc = 0
        for batch_x, target in zip(dataset['x'], dataset['y']):
            batch_x = batch_x.to(rbm.device)
            target = target.to(rbm.device)
            output = model(batch_x)
            loss = criterion(output, target)
            output = torch.argmax(output, dim=1)
            acc += torch.sum(output == target).item() / target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataset['y'])
        acc /= len(dataset['y'])
        progress.append(test(model, train_x, train_y, test_x, test_y, epoch + 1))
        training.set_description(str({'epoch': epoch + 1, 'loss': round(running_loss, 4), 'acc': round(acc, 4)}))

    return model, progress


if __name__ == '__main__':
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()

    vn = train_x.shape[1]
    hn = 2500
    rbm = RBM(vn, hn)
    rbm.load_rbm('../model/mnist_trained_rbm.pt')

    model = initialize_model()

    model, progress = train(model, train_x, train_y, train_x, train_y, test_x, test_y)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('../csv/RBM_without_pretraining_classifier.csv', index=False)

    model = initialize_model()

    model[0].weight = torch.nn.Parameter(rbm.W)
    model[0].bias = torch.nn.Parameter(rbm.hb)

    model, progress = train(model, train_x, train_y, train_x, train_y, test_x, test_y)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('../csv/RBM_pretrained_classifier.csv', index=False)
