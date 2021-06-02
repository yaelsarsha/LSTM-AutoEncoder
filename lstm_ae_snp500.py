import argparse
import math
import numpy as np
import pandas as pd
import pickle
import keras
from keras import layers
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from random import randrange
import matplotlib.pyplot as plt
from matplotlib.dates import (MONTHLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import matplotlib.ticker as ticker
import datetime
from time import gmtime, strftime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_size', type=int, default=3000, help='Dataset size')

parser.add_argument('--sample_length', type=int, default=300, help='Sample length')

parser.add_argument('--hidden_state_size', type=int, help='Hidden state size')

parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')

parser.add_argument('--optimizer', choices=['adam,sgd'], default='adam', help='Optimizer')

parser.add_argument('--lr', type=float, help='Learning rate')

parser.add_argument('--bs', type=int, help='Batch size')

args = parser.parse_args()

print('Train LSTM AutoEncoder on the Stock Prices Dataset',
      '\nDataset Size: ', args.dataset_size,
      '\nSample Length: ', args.sample_length,
      '\nHidden State Size: ', args.hidden_state_size,
      '\nNumber of epochs: ', args.epochs,
      '\nOptimizer: ', args.optimizer,
      '\nLearning rate: ', args.lr,
      '\nBatch size: ', args.bs)


def split_random_sequnces(target_size, target_length, train_set):
    split_train = np.zeros((target_size, target_length, train_set.shape[2]))
    indexes = np.zeros(target_size).astype(int)
    for i in range(target_size):
        index_sample = randrange(train_set.shape[0])
        index_start = randrange(train_set.shape[1] - target_length)
        short_sample = train_set[index_sample, index_start:index_start + target_length, :]
        split_train[i, :, :] = short_sample
        indexes[i] = index_start
    min_a = np.min(split_train, axis=1)
    max_a = np.max(split_train, axis=1)
    for i in range(target_size):
        split_train[i, :, :] = (split_train[i, :, :] - min_a[i, 0]) / (max_a[i, 0] - min_a[i, 0])
    return split_train, indexes


# SNP500 Data

stocks = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
df = stocks.pivot(index='symbol', columns='date', values='close').dropna(how='any', axis=0)

# Normalization and Centerization
full_dataset = df.subtract(df.min(axis=1), axis=0).divide(df.max(axis=1) - df.min(axis=1), axis=0).combine_first(df).values
full_length = full_dataset[0].shape[0]
sample_features = 1
full_size = full_dataset.shape[0]
full_dataset = full_dataset.reshape(full_size, full_length, sample_features)

# Split to Train, Test and Validation
full_train, full_validate, full_test = np.split(full_dataset, [int(.6 * len(full_dataset)), int(.8 * len(full_dataset))])

print('SNP500 Data-Set Size: ' + str(full_dataset.shape))

rule = rrulewrapper(MONTHLY, interval=4)
loc = RRuleLocator(rule)
formatter = DateFormatter('%d/%m/%y')

is_save = True
question1 = True
question2 = True
question3 = True

if question1:
    amazon_stock = stocks[stocks['symbol'] == 'AMZN']
    google_stock = stocks[stocks['symbol'] == 'GOOGL']

    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    amazon_stock.groupby('date')['high'].max().plot(ax=ax[0])
    ax[0].set_title('Daily Max Value of Amazon Stocks')
    google_stock.groupby('date')['high'].max().plot(ax=ax[1])
    ax[1].set_title('Daily Max Value of Google Stocks')
    plt.savefig('daily max value of stocks.png')

if question2:

    dataset_size = 10000  # args.dataset_size
    sample_length = 300  # args.sample_length

    train, train_i = split_random_sequnces(dataset_size, sample_length, full_train)
    validate, validate_i = split_random_sequnces(500, sample_length, full_validate)
    test, test_i = split_random_sequnces(500, sample_length, full_test)

    print('SNP500 Data Train-Set Size: ' + str(train.shape))
    print('SNP500 Data Validation-Set Size: ' + str(validate.shape))
    print('SNP500 Data Test-Set Size: ' + str(test.shape))
    sample_length = train[0].shape[0]
    dataset_size = train.shape[0]

    hidden_state_sizes = [args.hidden_state_size] if args.hidden_state_size is not None else [128, 256]
    batch_sizes = [args.bs] if args.bs is not None else [32, 64]
    learning_rates = [args.lr] if args.lr is not None else [0.005, 0.01]
    epochs = args.epochs
    optimizer = args.optimizer
    loss = 'mse'

    for hs in hidden_state_sizes:
        for bs in batch_sizes:
            for lr in learning_rates:
                test_name = "snp q2 " + str(dataset_size) + ' ' + str(sample_length) + ' ' + str(hs) + ' ' + str(bs) + ' ' + str(lr).replace(".", "_") + ' ' + optimizer + ' ' + loss
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
                history = sy_ae.fit(train, train, epochs=epochs, batch_size=bs, verbose=not is_save, validation_data=(validate, validate))

                print(test_name + "\nTrain Loss: " + "{:10.4f}".format(history.history['loss'][-1]) + "\nValidation Loss: " + "{:10.4f}".format(history.history['val_loss'][-1]))
                print(history.history)

                try:
                    sy_ae.save(test_name)
                except:
                    print("An exception occurred")

                # Loss Graph
                fig = plt.figure(figsize=(10, 5))
                plt.plot(np.concatenate([[1.0], history.history['loss']], axis=0), label='Train')
                plt.plot(np.concatenate([[1.0], history.history['val_loss']], axis=0), label='Validation')
                plt.legend()
                plt.title('Training Loss Graph')
                plt.ylabel('loss value')
                plt.xlabel('epochs')
                plt.yscale('log')
                if is_save:
                    plt.savefig('SNP500 data loss q2 ' + test_name + '.png')
                else:
                    plt.show()

                train_predict = sy_ae.predict(train)
                validate_predict = sy_ae.predict(validate)
                print(history.history)

                test_predict = sy_ae.predict(test)
                fig = plt.figure(figsize=(15, 3))
                fig.suptitle('SP-500 Predict Examples')
                i = 0
                print(str(train_i[i]) + ' ' + str(train_i[i] + sample_length) + ' ' + str(train_predict[i].shape) + ' ' + str(validate_i[i]) + ' ' + str(validate_i[i] + sample_length) + ' ' + str(validate_predict[i].shape) + ' ' + str(test_i[i]) + ' ' + str(test_i[i] + sample_length) + ' ' + str(test_predict[i].shape))
                print(df.columns[train_i[i]:train_i[i] + sample_length])
                print(train[i])
                print(train_predict[i])
                print(df.columns[validate_i[i]:validate_i[i] + sample_length])
                print(validate[i])
                print(validate_predict[i])
                print(df.columns[test_i[i]:test_i[i] + sample_length])
                print(test[i])
                print(test_predict[i])
                # Plot Train i-Sample
                ax = plt.subplot(1, 3, i + 1)
                ax.plot(pd.to_datetime(df.columns[train_i[i]:train_i[i] + sample_length]), train[i], label='$x$')
                ax.plot(pd.to_datetime(df.columns[train_i[i]:train_i[i] + sample_length]), train_predict[i], label='$\hat{x}$')
                ax.legend()
                ax.xaxis.set_major_locator(RRuleLocator(rrulewrapper(MONTHLY, interval=1)))
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_tick_params(rotation=30, labelsize=10)
                ax.set_title('Train Sample ' + str(i + 1))
                ax.set(xlabel='Time', ylabel='Value')

                # Plot Validation i-Sample
                ax = plt.subplot(1, 3, i + 2)
                ax.plot(pd.to_datetime(df.columns[validate_i[i]:validate_i[i] + sample_length]), validate[i], label=r'$x$')
                ax.plot(pd.to_datetime(df.columns[validate_i[i]:validate_i[i] + sample_length]), validate_predict[i], label=r'$\hat{x}$')
                ax.legend()
                ax.xaxis.set_major_locator(RRuleLocator(rrulewrapper(MONTHLY, interval=1)))
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_tick_params(rotation=30, labelsize=10)
                ax.set_title('Validation Sample ' + str(i + 1))
                ax.set(xlabel='Time', ylabel='Value')

                # Plot Test i-Sample
                ax = plt.subplot(1, 3, i + 3)
                ax.plot(pd.to_datetime(df.columns[test_i[i]:test_i[i] + sample_length]), test[i], label=r'$x$')
                ax.plot(pd.to_datetime(df.columns[test_i[i]:test_i[i] + sample_length]), test_predict[i], label=r'$\hat{x}$')
                ax.legend()
                ax.xaxis.set_major_locator(RRuleLocator(rrulewrapper(MONTHLY, interval=1)))
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_tick_params(rotation=30, labelsize=10)
                ax.set_title('Test Sample ' + str(i + 1))
                ax.set(xlabel='Time', ylabel='Value')

                fig.tight_layout()
                plt.subplots_adjust(top=0.8)
                if is_save:
                    plt.savefig('SNP500 data predict q2 ' + test_name + '.png')
                else:
                    plt.show()

if question3:

    dataset_size = args.dataset_size
    sample_length = args.sample_length

    train, train_i = split_random_sequnces(dataset_size, sample_length, full_train)
    validate, validate_i = split_random_sequnces(500, sample_length, full_validate)
    test, test_i = split_random_sequnces(500, sample_length, full_test)

    print('SNP500 Data Train-Set Size: ' + str(train.shape))
    print('SNP500 Data Validation-Set Size: ' + str(validate.shape))
    print('SNP500 Data Test-Set Size: ' + str(test.shape))
    sample_length = train[0].shape[0]
    dataset_size = train.shape[0]

    hidden_state_sizes = [args.hidden_state_size] if args.hidden_state_size is not None else [128, 256]
    batch_sizes = [args.bs] if args.bs is not None else [32, 64]
    learning_rates = [args.lr] if args.lr is not None else [0.005, 0.01]
    epochs = 300  # args.epochs
    optimizer = args.optimizer
    loss = 'mse'

    for hs in hidden_state_sizes:
        for bs in batch_sizes:
            for lr in learning_rates:
                try:
                    test_name = "snp q3 " + str(dataset_size) + ' ' + str(sample_length) + ' ' + str(hs) + ' ' + str(bs) + ' ' + str(lr).replace(".", "_") + ' ' + optimizer + ' ' + loss
                    # Encoder
                    input = layers.Input(shape=(sample_length - 1, sample_features))
                    encoder = layers.LSTM(hs, activation='tanh')(input)
                    # Reconstruct Decoder
                    decoder_r = layers.RepeatVector(sample_length - 1)(encoder)
                    decoder_r = layers.LSTM(hs, activation='tanh', return_sequences=True)(decoder_r)
                    decoder_r = layers.TimeDistributed(layers.Dense(sample_features))(decoder_r)
                    # Predict Decoder
                    decoder_p = layers.RepeatVector(sample_length)(encoder)
                    decoder_p = layers.LSTM(hs, activation='tanh', return_sequences=True)(decoder_p)
                    decoder_p = layers.TimeDistributed(layers.Dense(sample_features))(decoder_p)

                    model = Model(inputs=input, outputs=[decoder_r, decoder_p])
                    model.compile(optimizer=optimizer, loss=loss)
                    history = model.fit(train[:, 0:-1, :], [train[:, 0:-1, :], train], epochs=epochs, batch_size=bs, verbose=not is_save, validation_data=(validate[:, 0:-1, :], [validate[:, 0:-1, :], validate]))

                    print(test_name + "\nTrain Loss: " + "{:10.4f}".format(history.history['loss'][-1]) + "\nValidation Loss: " + "{:10.4f}".format(history.history['val_loss'][-1]))
                    # with open(test_name + strftime("%Y-%m-%d %H%M%S", gmtime()) + '.pkl', 'wb') as f:
                    #     pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
                    print(history.history)

                    try:
                        sy_ae.save(test_name)
                    except:
                        print("An exception occurred")

                    # Loss Graph
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(history.history['loss'], label='Train')
                    plt.plot(history.history['val_loss'], label='Validation')
                    plt.legend()
                    plt.title('Training Loss Graph')
                    plt.ylabel('loss value')
                    plt.xlabel('epochs')
                    plt.yscale('log')
                    if is_save:
                        plt.savefig('SNP500 data loss q3 ' + test_name + '.png')
                    else:
                        plt.show()

                        # Reconstruct Graph
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(history.history['val_time_distributed_loss'], label='Train')
                    plt.plot(history.history['time_distributed_loss'], label='Validation')
                    plt.legend()
                    plt.title('Reconstruct Loss Graph')
                    plt.ylabel('loss value')
                    plt.xlabel('epochs')
                    plt.yscale('log')
                    if is_save:
                        plt.savefig('SNP500 data loss r q3 ' + test_name + '.png')
                    else:
                        plt.show()

                        # Predict Graph
                    fig = plt.figure(figsize=(10, 5))
                    plt.plot(history.history['val_time_distributed_1_loss'], label='Train')
                    plt.plot(history.history['time_distributed_1_loss'], label='Validation')
                    plt.legend()
                    plt.title('Predict Loss Graph')
                    plt.ylabel('loss value')
                    plt.xlabel('epochs')
                    plt.yscale('log')
                    if is_save:
                        plt.savefig('SNP500 data loss p q3 ' + test_name + '.png')
                    else:
                        plt.show()
                except:
                    print("An exception occurred")