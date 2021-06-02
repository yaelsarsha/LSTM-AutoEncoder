import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
from keras import backend as K
import pickle
import datetime
from time import gmtime, strftime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--hidden_state_size', type=int, help='Hidden state size')

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

parser.add_argument('--optimizer', choices=['adam,sgd'], default='adam', help='Optimizer')

parser.add_argument('--lr', type=float, help='Learning rate')

parser.add_argument('--bs', type=int, help='Batch size')

args = parser.parse_args()

print('Train LSTM AutoEncoder on the MNIST Dataset',
    '\nHidden State Size: ', args.hidden_state_size,
    '\nNumber of epochs: ', args.epochs,
    '\nOptimizer: ', args.optimizer,
    '\nLearning rate: ', args.lr,
    '\nBatch size: ', args.bs)

# MNIST Data

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Split to Test and Validation
x_validation, x_test = np.split(x_test, 2)
y_validation, y_test = np.split(y_test, 2)

image_weight = 28
long_seq = False
is_save = True
sample_length = image_weight if not long_seq else image_weight * image_weight
sample_features = image_weight if not long_seq else 1

x_train = x_train.reshape((len(x_train), sample_length, sample_features))
x_validation = x_validation.reshape((len(x_validation), sample_length, sample_features))
x_test = x_test.reshape((len(x_test), sample_length, sample_features))

print('MNIST Data Train-Set Size: ' + str(x_train.shape))
print('MNIST Data Validation-Set Size: ' + str(x_validation.shape))
print('MNIST Data Test-Set Size: ' + str(x_test.shape))

question1 = False
question2 = True

if question1:
    hidden_state_sizes = [args.hidden_state_size] if args.hidden_state_size is not None else [64, 128]
    batch_sizes = [args.bs] if args.bs is not None else [32, 64]
    learning_rates = [args.lr] if args.lr is not None else [0.05, 0.01, 0.005]
    epochs = args.epochs
    optimizer = args.optimizer
    loss = 'mse'

    for hs in hidden_state_sizes:
        for bs in batch_sizes:
            for lr in learning_rates:
                test_name = str(hs) + ' ' + str(bs) + ' ' + str(lr).replace(".", "_") + ' ' + optimizer + ' ' + loss
                sy_ae = Sequential(
                    [
                        # Encoder
                        layers.LSTM(units=hs, return_sequences=False, activation='tanh'),
                        layers.RepeatVector(sample_length),
                        # Decoder
                        layers.LSTM(units=hs, return_sequences=True, activation='tanh'),
                        layers.TimeDistributed(layers.Dense(sample_features, activation='tanh'))
                    ]
                )

                sy_ae.compile(optimizer=optimizer, loss=loss)
                K.set_value(sy_ae.optimizer.learning_rate, lr)
                history = sy_ae.fit(x_train, x_train, epochs=epochs, batch_size=bs, verbose=not is_save, validation_data=(x_validation, x_validation))
                print(test_name + "\nTrain Loss: " + "{:10.4f}".format(history.history['loss'][-1]) + "\nValidation Loss: " + "{:10.4f}".format(history.history['val_loss'][-1]))
                print(history.history)

                # Loss Graph
                fig = plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Train')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.title('Training Loss Graph')
                plt.ylabel('loss value')
                plt.xlabel('epoch')
                # plt.yscale('log')
                if is_save:
                    plt.savefig('mnist data loss graph q1 ' + test_name + '.png')
                else:
                    plt.show()

                train_predict = sy_ae.predict(x_train)
                test_predict = sy_ae.predict(x_test)
                print(train_predict[1].shape)

                n = 10  # How many digits we will display
                fig = plt.figure(figsize=(20, 4))
                for i in range(n):
                    # Display original
                    ax = plt.subplot(2, n, i + 1)
                    plt.imshow(x_test[i].reshape(image_weight, image_weight))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                    # Display reconstruction
                    ax = plt.subplot(2, n, i + 1 + n)
                    plt.imshow(test_predict[i].reshape(image_weight, image_weight))
                    plt.gray()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

                if is_save:
                    plt.savefig('mnist data predict q1 ' + test_name + '.png')
                else:
                    plt.show()

if question2:
    y_train = keras.utils.to_categorical(y_train)
    y_validation = keras.utils.to_categorical(y_validation)
    y_test = keras.utils.to_categorical(y_test)

    hidden_state_sizes = [args.hidden_state_size] if args.hidden_state_size is not None else [64, 128]
    batch_sizes = [args.bs] if args.bs is not None else [32, 64]
    learning_rates = [args.lr] if args.lr is not None else [0.05, 0.01, 0.005]
    epochs = args.epochs
    optimizer = args.optimizer
    loss = 'binary_crossentropy'

    for hs in hidden_state_sizes:
        for bs in batch_sizes:
            for lr in learning_rates:
                test_name = 'dropout ' + str(hs) + ' ' + str(bs) + ' ' + str(lr).replace(".", "_") + ' 0_2 ' + optimizer + ' ' + loss
                sy_ae = Sequential(
                    [
                        # Encoder
                        layers.LSTM(units=hs, return_sequences=False, activation='tanh', recurrent_dropout=0.2),
                        layers.RepeatVector(sample_length),
                        # Decoder
                        layers.LSTM(units=hs, recurrent_dropout=0.2),
                        layers.Dense(10, activation='sigmoid')
                    ]
                )

                sy_ae.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                history = sy_ae.fit(x_train, y_train, epochs=epochs, batch_size=bs, verbose=0, validation_data=(x_validation, y_validation))

                print(test_name + "\nTrain Loss: " + "{:10.4f}".format(history.history['loss'][-1]) + "\nValidation Loss: " + "{:10.4f}".format(history.history['val_loss'][-1]) + \
                      "\nTrain Accuracy: " + "{:10.2f}".format(history.history['accuracy'][-1] * 100) + "%\nValidation Accuracy: " + "{:10.2f}".format(history.history['val_accuracy'][-1] * 100) + '%\n')
                print(history.history)

                # Loss Graph
                fig = plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Train')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.title('Training Loss Graph')
                plt.ylabel('loss value')
                plt.xlabel('epoch')
                # plt.yscale('log')
                if is_save:
                    plt.savefig('mnist data loss q2 ' + test_name + '.png')
                else:
                    plt.show()

                # Accuracy Graph
                fig = plt.figure(figsize=(10, 5))
                plt.plot(history.history['accuracy'], label='Train')
                plt.plot(history.history['val_accuracy'], label='Validation')
                plt.legend()
                plt.title('Training Accuracy Graph')
                plt.ylabel('accuracy value')
                plt.xlabel('epoch')
                plt.yscale('log')
                if is_save:
                    plt.savefig('mnist data accuracy q2 ' + test_name + '.png')
                else:
                    plt.show()

                predict = sy_ae.predict(x_test)
                fig = plt.figure(figsize=(10, 5))
                for j in range(10):
                    plt.subplot(2, 5, j + 1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(x_test[j], cmap=plt.cm.binary)
                    plt.xlabel(np.argmax(predict[j]))

                if is_save:
                    plt.savefig('mnist data predict q2 ' + test_name + '.png')
                else:
                    plt.show()
