import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
from keras import backend as K

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_size', type=int, default=10000, help='Input size')

parser.add_argument('--hidden_state_size', type=int, help='Hidden state size')

parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')

parser.add_argument('--optimizer', choices=['adam,sgd'], default='adam', help='Optimizer')

parser.add_argument('--lr', type=float, help='Learning rate')

parser.add_argument('--bs', type=int, help='Batch size')

parser.add_argument('--gc', type=int, help='Gradient clipping')

args = parser.parse_args()

print('Train LSTM AutoEncoder on the Synthetic Dataset',
    '\nInput Size: ', args.input_size,
    '\nHidden State Size: ', args.hidden_state_size,
    '\nNumber of epochs: ', args.epochs,
    '\nOptimizer: ', args.optimizer,
    '\nLearning rate: ', args.lr,
    '\nBatch size: ', args.bs,
    '\nGradient clipping: ', args.gc)

# Generate Synthetic Data
sample_length = 50
sample_features = 1
dataset_size = args.input_size
synthetic_data_set = np.random.rand(dataset_size, sample_length)

# Normalization and Centerization
synthetic_data_set = (synthetic_data_set - np.array([np.amin(synthetic_data_set, axis=1)]).T) / (np.array([np.amax(synthetic_data_set, axis=1)]).T - np.array([np.amin(synthetic_data_set, axis=1)]).T)
synthetic_data_set = synthetic_data_set + (0.5 - np.array([np.mean(synthetic_data_set, axis=1)]).T)

synthetic_data_set = np.reshape(synthetic_data_set, (dataset_size, sample_length, sample_features))

# Split to Train,Test and Validation
train, validate, test = np.split(synthetic_data_set, [int(.6 * len(synthetic_data_set)), int(.8 * len(synthetic_data_set))])

print('Synthetic Data Train-Set Size: ' + str(train.shape))
print('Synthetic Data Validation-Set Size: ' + str(validate.shape))
print('Synthetic Data Test-Set Size: ' + str(test.shape))

times = range(1, sample_length + 1)
i = 0
is_save = True
question1 = True
question2 = True

if question1:
    fig = plt.figure(figsize=(15, 3))
    fig.suptitle('Synthetic Data Examples')

    # Plot Train i-Sample
    ax = plt.subplot(1, 2, i + 1)
    ax.plot(times, train[i], label='$x$')
    ax.legend()
    ax.set_title('Train Sample ' + str(i + 1))
    ax.set(xlabel='Time', ylabel='Value')

    # Plot Test i-Sample
    ax = plt.subplot(1, 2, i + 2)
    ax.plot(times, test[i], label=r'$x$')
    ax.legend()
    ax.set_title('Test Sample ' + str(i + 1))
    ax.set(xlabel='Time', ylabel='Value')

    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    if is_save:
        plt.savefig('synthetic data examples q1.png')
    else:
        plt.show()

if question2:
    hidden_state_sizes = [args.hidden_state_size] if args.hidden_state_size is not None else [64, 128, 256]
    batch_sizes = [args.bs] if args.bs is not None else [32, 64]
    learning_rates = [args.lr] if args.lr is not None else [0.05, 0.01, 0.005]
    clip_values = [args.gc] if args.gc is not None else [0.5, 1]
    epochs = args.epochs
    optimizer = args.optimizer
    loss = 'mse'

    for hs in hidden_state_sizes:
        for bs in batch_sizes:
            for lr in learning_rates:
                for cv in clip_values:
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
                    #   K.clip(r, -eps, eps)
                    #   K.set_value(sy_ae.optimizer.clipvalue, cv)
                    history = sy_ae.fit(train, train, epochs=epochs, batch_size=bs, verbose=not is_save, validation_data=(validate, validate))

                    # Loss Graph
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(history.history['loss'], label='Train')
                    plt.plot(history.history['val_loss'], label='Validation')
                    plt.legend()
                    plt.title('Training Loss Graph')
                    plt.ylabel('loss value')
                    plt.xlabel('epochs')
                    plt.yscale('log')
                    plt.show()
                    plt.savefig('synthetic data loss graph ' + test_name + '.png')

                    train_predict = sy_ae.predict(train)
                    test_predict = sy_ae.predict(test)
                    print(train_predict[1].shape)

                    # Predict Graph
                    fig = plt.figure(figsize=(15, 3))
                    fig.suptitle('Synthetic Data Predict Example')

                    # Plot Train i-Sample
                    ax = plt.subplot(1, 2, i + 1)
                    ax.plot(times, train[i], label='$x$')
                    ax.plot(times, train_predict[i], label='$\hat{x}$')
                    ax.legend()
                    ax.set_title('Train Sample ' + str(i + 1))
                    ax.set(xlabel='Time', ylabel='Value')

                    # Plot Test i-Sample
                    ax = plt.subplot(1, 2, i + 2)
                    ax.plot(times, test[i], label=r'$x$')
                    ax.plot(times, test_predict[i], label=r'$\hat{x}$')
                    ax.legend()
                    ax.set_title('Test Sample ' + str(i + 1))
                    ax.set(xlabel='Time', ylabel='Value')

                    fig.tight_layout()
                    plt.subplots_adjust(top=0.8)
                    if is_save:
                        plt.savefig('synthetic data predict ' + test_name + '.png')
                    else:
                        plt.show()