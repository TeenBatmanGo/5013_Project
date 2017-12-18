#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 5th 2017

@author Lingxiao Zhang, id#20475043
        Chengming Wang, id#20450392
        Guanlan Lu, id#20454477
        Yilan Yan， id#20475213

"""
import numpy as np
import talib as tb

# How long to perform an analysis and transaction
time_span = 60

# For MACD only
fast_span = 12
slow_span = 26
signal_span = 9

# For KD only
kd_low = 10
kd_high = 90
fastk_period = 5
slowk_period = 3
slowd_period = 3


# Rate of the initial cash to perform transactions
cost_rate = 0.04

# Target indexes
asset_index1 = [1, 6]
asset_index2 = [0]
#asset_index =[12, 10, 6, 3]
#asset_index = [11, 10]
#asset_index =[8]

# Weight of each index on the cash
weight_index1 = [1.4, 0.8]
weight_index2 = [0.9]
#weight_index = [1.1, 1.1, 0.7, 0.35]
#weight_index=[1.5]
#weight_index=[1.2, 1.2]

# Cutting-loss criterion
my_cash_balance_lower_limit = 2000000 


# The strategy function that performs future trading on the given dataset
def handle_bar(timer, data, info, init_cash, transaction, detail_last_min, memory):

    # Get position of last minute (size 13x1)
    position_new = detail_last_min[0]  # latest position matrix

    # Initialize all variables
    if timer == 0:
       memory.totalCounter = 0
       memory.spanCounter = 0

       # Whole dataset from the beginning to the end
       memory.data_list = list()

    # Do the analysis here
    if memory.spanCounter % time_span == 0 and memory.spanCounter != 0 and detail_last_min[1] > my_cash_balance_lower_limit:
        # For each index
        for i in range(len(asset_index1)):
            # All of the current data of the given index
            data1 = generate_data_helper(memory.data_list, asset_index1[i])
            # Calculate the average price of
            average_price = compute_average(data1)
            # myDif --> macd fast moving average   myDea --> slow moving average
            myDif, myDea, myBAr = tb.MACD(average_price, fastperiod=fast_span, slowperiod=slow_span,
                                      signalperiod=signal_span)

            current_average_sum = compute_average(data)
            current_average_target = current_average_sum[asset_index1[i]]
            lot_value = current_average_target * info.unit_per_lot[asset_index1[i]] * info.margin_rate[asset_index1[i]]
            num_lot = np.round(cost_rate*(weight_index1[i])*init_cash/(lot_value*(1.+transaction)))

            # Make the decision
            if myDif[-1] - myDif[-2] > 0 and myDif[-1] > myDea[-1] and myDif[-1] - myDif[-3] >0 : # Buy
               position_new[asset_index1[i]] += num_lot
            elif myDif[-1] - myDif[-2] < 0 and myDif[-1] < myDea[-1] and myDif[-1] - myDif[-3] <0: # Sell
               position_new[asset_index1[i]] -= num_lot
            else:
               pass
           
        if detail_last_min[1] > my_cash_balance_lower_limit:
            for j in range(len(asset_index2)):
                data1 = generate_data_helper(memory.data_list, asset_index2[j])
                slowk, slowd = tb.STOCH(data1[:, 1], data1[:, 2], data1[:, 3],
                                    fastk_period=fastk_period, slowk_period=slowk_period
                                    , slowk_matype =0, slowd_period=slowd_period, slowd_matype=0)
                current_average_target = current_average_sum[asset_index2[j]]
                lot_value = current_average_target * info.unit_per_lot[asset_index2[j]] * info.margin_rate[asset_index2[j]]
                num_lot = np.round(cost_rate*(weight_index2[j])*init_cash/(lot_value*(1.+transaction)))
    
                if slowd[-1] < kd_low or slowk[-1] < kd_low:
                   position_new[asset_index2[j]] += num_lot
                elif slowd[-1] > kd_high or slowk[-1] > kd_high:
                   position_new[asset_index2[j]] -= num_lot
                else:
                   pass

    # Update values
    memory.data_list.append(data)
    memory.totalCounter += 1
    memory.spanCounter += 1
    return position_new, memory


# Returns all data of a given item
# Row -- 1 day
# numbers of days x 4
def generate_data_helper(data_list, asset):
    asset_open = np.array([array[asset, 0] for array in data_list])
    asset_high_price = np.array([array[asset, 1] for array in data_list])
    asset_low_price = np.array([array[asset, 2] for array in data_list])
    asset_close_price = np.array([array[asset, 3] for array in data_list])

    assetData = np.array([asset_open, asset_high_price, asset_low_price, asset_close_price]).T

    return assetData


# Returns the given few days of the dataset
# Given the number of past minutes
def get_past_data(data_input, numPastmins):
    if (numPastmins > data_input.shape[0]):
        raise ValueError ("number exceeds")
    else:
        return data_input[data_input.shape[0]-numPastmins:]



# Returns the average of the given dataset on each row
def compute_average(data_input):
     average = np.mean(data_input, axis=1)
     return average



# Update position matrix
# Given amount && index
def update_position(current_position, amount, index):
    return current_position[index] == amount

if __name__ == '__main__':
    ''' This strategy simply check if there is any special technical pattern in data
        No training process required. Main function is passed.
    '''
