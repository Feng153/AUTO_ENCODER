import numpy as np
import torch
from dbn import DBN

from data_charts.load_dataset import MNIST
from tqdm import trange
import pandas as pd


def initialize_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.Sigmoid(),
        torch.nn.Linear(512, 128),
        torch.nn.Sigmoid(),
        torch.nn.Linear(128, 64),
        torch.nn.Sigmoid(),
        torch.nn.Linear(64, 10),
        torch.nn.Softmax(dim=1),
    )
    model.cuda(dbn.device)
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
    test_x1 = test_x.to(dbn.device)
    test_y1 = test_y.to(dbn.device)
    train_y1 = train_y.to(dbn.device)
    train_x1 = train_x.to(dbn.device)

    output_test = model(test_x1)
    loss_test = criterion(output_test, test_y1).item()
    output_test = torch.argmax(output_test, axis=1)
    acc_test = torch.sum(output_test == test_y1).item() / test_y1.shape[0]

    output_train = model(train_x1)
    loss_train = criterion(output_train, train_y1).item()
    output_train = torch.argmax(output_train, axis=1)
    acc_train = torch.sum(output_train == train_y1).item() / train_y1.shape[0]
    return epoch, loss_test, loss_train, acc_test, acc_train


def train(model, x, y, train_x, train_y, test_x, test_y, epoch=5):
    dataset = generate_batches(x, y)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    training = trange(epoch)
    progress = []
    for epoch in training:
        running_loss = 0
        acc = 0
        for batch_x, target in zip(dataset['x'], dataset['y']):
            batch_x = batch_x.to(dbn.device)
            target = target.to(dbn.device)
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

    layers = [512, 128, 64, 10]

    dbn = DBN(train_x.shape[1], layers, savefile='../model/mnist_trained_dbn.pt')
    dbn.train_DBN(train_x)

    model = dbn.initialize_model()

    completed_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    completed_model.to(dbn.device)
    torch.save(completed_model, '../model/mnist_trained_dbn_classifier.pt')

    print(completed_model)

    print('\n' * 3)
    print("Without Pre-Training")
    model = initialize_model()
    model, progress = train(model, train_x, train_y, train_x, train_y, test_x, test_y)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('../csv/DBN_without_pretraining_classifier.csv', index=False)

    print("With Pre-Training")
    model, progress = train(completed_model, train_x, train_y, train_x, train_y, test_x, test_y)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('../csv/DBN_with_pretraining_classifier.csv', index=False)

    print("With Pre-Training and Original Images turned Binary")

    train_x = train_x > torch.mean(train_x)
    test_x = test_x > torch.mean(test_x)
    train_x, test_x = train_x.int().float(), test_x.int().float()

    model, progress = train(completed_model, train_x, train_y, train_x, train_y, test_x, test_y)
    progress = pd.DataFrame(np.array(progress))
    progress.columns = ['epochs', 'test loss', 'train loss', 'test acc', 'train acc']
    progress.to_csv('../csv/DBN_with_pretraining_and_input_binarization_classifier.csv', index=False)
