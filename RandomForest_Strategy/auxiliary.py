import h5py
import numpy as np
import pandas as pd
import talib
from readdata import read_h5
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


stock = 10
window_size = 30
prev = 5


dicts = read_h5('/Users/wangchengming/Documents/5013Project/MSBD5013/pythonplatform/Data/data_format1_20170717_20170915.h5')
keys = list(dicts.keys())
data = dicts[keys[stock]]
stoch = talib.STOCHF(data.high.values, data.low.values, data.close.values)[0][14:]
rsi = talib.RSI(data.close.values)[14:]
rolling_mean = data.close.rolling(window=window_size).mean()
obv = talib.OBV(data.close.values, data.volume.values)[14:]
featdata = data.iloc[14:, -2:].assign(stoch=stoch, rsi=rsi, rolling=rolling_mean, obv=obv)


def add_label_2(ori_data, data, previous=5):
    back = pd.DataFrame(ori_data.iloc[:, -2].shift(previous))
    forward = pd.DataFrame(ori_data.iloc[:, -2].shift(-1))
    merged = pd.merge(back, forward, left_index=True, right_index=True).dropna()
    merged = merged.assign(label=lambda x: np.where((x.iloc[:, -1]-x.iloc[:, 0])>=0, 1, 0))
    merged = merged.drop(['close_x', 'close_y'], axis=1)
    newdata = pd.merge(data, merged, left_index=True, right_index=True).dropna()
    return newdata


labelled_data = add_label_2(data, featdata, prev)
clf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=0)
clf.fit(labelled_data.iloc[:, :-1].values, labelled_data.label.values)

joblib.dump(clf, 'randomforest2.pkl')