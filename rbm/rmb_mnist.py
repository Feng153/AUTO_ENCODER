import torch

from data_charts.load_dataset import MNIST
from rbm import RBM

if __name__ == '__main__':
    mnist = MNIST()
    train_x, train_y, test_x, test_y = mnist.load_dataset()
    print('MAE for all 0 selection:', torch.mean(train_x))
    vn = train_x.shape[1]
    hn = 2500

    rbm = RBM(vn, hn, epochs=20, mode='bernoulli', lr=0.0005, k=5, batch_size=128, gpu=True, optimizer='adam',
              savefile='../model/mnist_trained_rbm.pt', early_stopping_patience=5)
    rbm.train(train_x)
