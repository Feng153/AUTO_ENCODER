import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

np.random.seed(33)

ENCODING_DIM_INPUT = 784
ENCODING_DIM_OUTPUT = 2
EPOCHS = 20
BATCH_SIZE = 64


def train(x_train):
    """
    build autoencoder
    :param x_train: train_data
    :return: encoder and decoder
    """
    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT,))

    # encoding layer
    hidden_layer = Dense(ENCODING_DIM_OUTPUT, activation='relu')(input_image)

    # decoding layer
    decode_output = Dense(ENCODING_DIM_INPUT, activation='relu')(hidden_layer)

    # build autoencoder, encoder, decoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=hidden_layer)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    return encoder, autoencoder


def plot_representation(encode_image, y_test):
    """
    plot the hidden result
    :param encode_image: images after encoding
    :param y_test: label
    :return:
    """
    plt.scatter(encode_image[:, 0], encode_image[:, 1], c=y_test, s=3)
    plt.colorbar()
    plt.show()


def show_image(decode_images, x_test):
    """
    plot the images
    :param decode_images: images after decoding
    :param x_test: testing data
    :return:
    """
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # reshape data
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    # train
    encoder, autoencoder = train(x_train=x_train)
    # test and plot
    encode_images = encoder.predict(x_test)
    plot_representation(encode_images, y_test)

    # show images
    decode_images = autoencoder.predict(x_test)
    show_image(decode_images, x_test)


