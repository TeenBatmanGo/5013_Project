#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:27:48 2017

@author: MAngO
"""

import numpy as np
from sample.auxiliary import *
import os
from keras.models import load_model
import talib

interval = 15
asset_index = 0
my_cash_balance_lower_limit = 3000000

model_dir = '/Users/wangchengming/Documents/HKUST/5013/MSBD5013/pythonplatform/'
model = load_model(model_dir + 'LSTM_15_model.h5')


def handle_bar(timer, data, info, init_cash, transaction, detail_last_min, memory):
    ''' Write your own strategy here in the strategy function

    Params: timer = int, counter of current time
            data = pandas.dataframe, data for current minute bar
            info - pandas.dataframe, information matrix
            init_cash，transaction - double, constans
            detail_last_min - list, contains cash balance, margin balance, total balance and position of last minute
            memory - class, current memory of your strategy
    
    You are allowed to access the data(open, high, low, close, volume) of current minute,
    the current time, initial settings(initial cash and transaction cost).
    
    Notes:
    1. All data else you need in your model and strategy should be stored by yourself in 
    memory variable.
    
    2. Return value are limited to be in the following form: position_matrix_of_next_minute, 
    memory_list
    
    3. Backtest module will accept return values from your strategy function and use them as
    new input into your strateg function in next minute.
    
    4. Strategy functions that cannot operate properly in back test may ower your final grade.
    Please double check to make sure that your strategy function satisfy all the requirments above.
    '''

    position_new = detail_last_min[0]

    if (timer == 0):
        memory.model = model
        memory.temp_data_list = list()
        memory.data_list = list()

    if ((timer+1)%interval == 0 and timer!=0):
        memory.temp_data_list.append(data)
        memory.data_list.append(np.mean(np.array(memory.temp_data_list), axis=0))
        sto = -1
        # Stachastic Ostilator based on dataset with 15 minutes average
        if (timer+1 >= interval*9):
            df = np.array([dat[asset_index] for dat in memory.data_list])
            sto_li = talib.STOCH(df[:, 1], df[:, 2], df[:, 3])[0]
            sto = [st for st in sto_li if st is not np.nan][-1]
        #df = np.array([dat[asset_index] for dat in memory.temp_data_list])
        #sto = talib.STOCH(df[:, 1], df[:, 2], df[:, 3])[0]
        #sto = [st for st in sto if st is not np.nan][-1]
        close = [dat[asset_index, 3] for dat in memory.temp_data_list]
        memory.temp_data_list = list()

        assert len(close)==interval, 'not enough close prices'

        series = np.array([ave(close, i, interval) for i in range(0, len(close), interval)]).reshape((-1, 1))
        test = scaling(series)
        preds = model.predict(test.reshape((1, 1, 1)), batch_size=1)
        preds = unscaling(preds).tolist()[0]

        if (preds[-1]-preds[0])/preds[0] >= 0.004 and sto < 20:
            if detail_last_min[1] > my_cash_balance_lower_limit:
                print('----------------Buy 50')
                position_new[asset_index] += 50

        elif (preds[-1]-preds[0])/preds[0] <= -0.004 and (sto > 80 or sto == -1):
            if detail_last_min[1] > my_cash_balance_lower_limit:
                print('----------------Sell 50')
                position_new[asset_index] -= 50

        elif (0.0001 < (preds[-1]-preds[0])/preds[0] < 0.004) and sto < 50:
            if detail_last_min[1] > my_cash_balance_lower_limit:
                print('----------------Buy 30')
                position_new[asset_index] += 30

        elif (-0.004 < (preds[-1]-preds[0])/preds[0] < -0.0001) and (sto > 50 or sto == -1):
            if detail_last_min[1] > my_cash_balance_lower_limit:
                print('----------------Sell 30')
                position_new[asset_index] -= 30

    else:
        memory.temp_data_list.append(data)
        memory.data_list.append(data)
    return position_new, memory





if __name__ == '__main__':
    pass