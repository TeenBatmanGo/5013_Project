from sklearn import svm
from strategy import generate_data_helper, compute_average, get_past_data
import talib as tb
import  numpy as np
import os
from sklearn.externals import joblib

time_span = 100 # how long to perform an analysis and transaction

# For MACD only
fast_span = 12
slow_span = 26
signal_span = 9


# For RSI only
rsi_lower = 40
rsi_higher = 60
rsi_period = 21


# For KD only
kd_low = 20
kd_high = 80
fastk_period = 5
slowk_period = 3
slowd_period = 3


cost_rate = 0.005


asset_index = [12, 10, 8, 3]
weight_index = [1.2, 1.2, 0.6, 0.4]

my_cash_balance_lower_limit = 4000000. # Cutting-loss criterion


# model_dir = '/Users/lingxiaozhang/Desktop/PythonPlatform1/demo3'
# os.chdir(model_dir)


#  memory.counter - 1 = timer
def handle_bar(timer, data, info, init_cash, transaction, detail_last_min, memory):


    element_array = ["A.DCE", "AG.SHF", "AU.SHF", "I.DCE", "IC.CFE", "IF.CFE", "IH.CFE",
                     "J.DCE", "JM.DCE", "M.DCE", "RB.SHF", "Y.DCE", "ZC.CZC"]

    # Get position of last minute
    # Size 13x1
    position_new = detail_last_min[0]  # latest position matrix

    # Initialize
    if timer == 0:
       memory.totalCounter = 0
       memory.spanCounter = 0

       memory.data_list = list()

       memory.bufferLabel = list()
       memory.bufferLabel.append(0)

       memory.bar_counter = 0
       memory.data_list = list() # whole dataset from the start to the end
       memory.flagSVM = []
       #memory.weight = pd.readcsv()

       memory.flagBuy = 0
       memory.flagSell = 0

    # Do the analysis here
    if memory.spanCounter % time_span == 0 and memory.spanCounter != 0 and detail_last_min[1] > my_cash_balance_lower_limit:
        for i in range(len(asset_index)):
            data1 = generate_data_helper(memory.data_list, asset_index[i])
            #data_close = data[:, -1]
            current_close = data[asset_index[i], 3]
            current_open = data[asset_index[i], 0]
            average_price = compute_average(data1)

            train_input = np.array([average_price[-1], current_close, current_open]).reshape(1, 3)
            # print average_price[-1]
            # print current_close
            # print train_input
            # print train_input.shape

            # DO SVM!!!
            path = 'svmindex' + str(asset_index[i]) + '.pkl'
            clf = joblib.load(path)
            memory.flagSVM = clf.predict(train_input)

            slowk, slowd = tb.STOCH(data1[:, 1], data1[:, 2], data1[:, 3],
                                    fastk_period=fastk_period, slowk_period=slowk_period
                                    , slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)


            # myDif --> macd fast moving average   myDea --> slow moving average
            myDif, myDea, myBAr = tb.MACD(average_price, fastperiod=fast_span, slowperiod=slow_span,
                                      signalperiod=signal_span)

            current_average_sum = compute_average(data)
            current_average_target = current_average_sum[asset_index[i]]


            lot_value = current_average_target * info.unit_per_lot[asset_index[i]] * info.margin_rate[asset_index[i]]
            num_lot = np.round(cost_rate*weight_index[i]*init_cash/(lot_value*(1.+transaction)))


            # MACD
            #if myDif[-1] - myDif[-2] > 0 and myDif[-1] > myDea[-1]:
             #  memory.flagBuy += 1
            #elif myDif[-1] - myDif[-2] < 0 and myDif[-1] < myDea[-1]:
             #  memory.flagSell += 1
            #else:
             #  pass

             # KD
            if slowk[-1] < kd_low or slowd[-1] < kd_low:
                memory.flagBuy += 1
            elif slowd[-1] > kd_high or slowk[-1] > kd_high:
                memory.flagSell += 1
            else:
                pass


            # Take actions
            #if memory.flagBuy >= 1 and memory.flagSVM == 1:
              # position_new[asset_index[i]] += num_lot
            #elif memory.flagSell >= 1 and memory.flagSVM == 0:
              # pass
            #elif memory.flagSell >= 1:
              # position_new[asset_index[i]] -= num_lot
            #else:
              # pass

            if memory.flagSVM == 1:
                memory.flagBuy += 1
            elif memory.flagSVM == 0:
                memory.flagSell += 1
            else:
                pass


            if memory.flagSell >= 1:
                position_new[asset_index[i]]-= int(num_lot/4)
            elif memory.flagBuy >= 1:
                position_new[asset_index[i]] += int(num_lot/4)
            else:
                pass


            memory.bufferLabel = list()
            memory.bufferLabel.append(0)
            memory.flagSVM = 0


    memory.data_list.append(data)
    memory.totalCounter += 1
    memory.spanCounter += 1
        # Reset
    memory.flagBuy = 0
    memory.flagSell = 0

    return position_new, memory

'''
def addLabel(data):
    train_label = list()
    for i in range(0, len(data)-1):
        if data[i+1]>data[i]:
            train_label.append(1)
        else:
            train_label.append(0)
    return train_label
'''



if __name__ == '__main__':
    ''' This strategy simply check if there is any special technical pattern in data
        No training process required. Main function is passed.
    '''
    pass