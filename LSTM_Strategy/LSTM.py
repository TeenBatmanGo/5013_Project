
# This LSTM model was pre-trained and stored in LSTM_model.h5.
# No need to run this file again.

# The basic idea of this LSTM approach is multiple time steps prediction to find future trend.
# Inputs are current time step data and outputs are data several time steps afterwards.

import h5py
import numpy as np
import pandas as pd
import os
from sample.auxiliary import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


asset_index = 0  # Index of the asset. Here it's A.DCE
interval = 15  # 15 minutes average to smooth the dataset
n_lag = 1  # Number of time steps as input
n_seq = 3  # Number of time steps as output


directory = '/Users/wangchengming/Documents/5013Project/MSBD5013/pythonplatform'
os.chdir(directory)


# Training dataset
dicts = read_h5('Data/data_format1_20170717_20170915.h5')
keys = list(dicts.keys())
data = dicts[keys[0]]


# Transform series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


# Use close data as series
series = data.close.values
# Average over certain minutes. Here it's 15 minutes.
series = pd.Series([ave(series, i, interval) for i in range(0, len(series), interval)])


def scale_data(series, n_test, n_lag, n_seq):
    vals = series.values
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(vals.reshape(len(vals), 1))
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    train = supervised.values
    return scaler, train


# Store the scaler in scaler.json
scaler, train = scale_data(series, n_lag, n_seq)
# sca = np.array([scaler.data_min_[0], scaler.data_max_[0], scaler.data_range_[0]]).reshape((1, -1))
# sca = pd.DataFrame(sca, columns=['min', 'max', 'range'])
# sca.to_json('scaler.json')


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True, \
                   return_sequences=False))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
        model.reset_states()
    return model

# This LSTM network contains one layer with 10 neurons.
# Store model in LSTM_model.h5
model = fit_lstm(train, 1, 3, 1, 100, 10)
# model.save('LSTM_15_model.h5')







