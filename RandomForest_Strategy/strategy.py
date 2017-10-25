
import numpy as np
from RandomForest.auxiliary import *
import os
from sklearn.externals import joblib
import talib


asset_index = [10, 12]

fast_span = 12
slow_span = 26
signal_span = 9

cost_rate = 0.035
weight_index = [1, 1]
my_cash_balance_lower_limit = 3000000

model_dir = '/Users/wangchengming/Documents/5013Project/MSBD5013/pythonplatform/'
os.chdir(model_dir)



def handle_bar(timer, data, info, init_cash, transaction, detail_last_min, memory):

    position_new = detail_last_min[0]

    if (timer == 0):
        memory.data_list = list()


    if timer % 30 == 0:
        memory.data_list.append(data)

        for i in range(len(asset_index)):
            ind = asset_index[i]
            weight = weight_index[i]
            path = 'randomforest' + str(i) + '.pkl'
            clf = joblib.load(path)
            subdata = np.array([dat[ind, :] for dat in memory.data_list])
            stoch = talib.STOCH(subdata[:, 1], subdata[:, 2], subdata[:, 3])[0][14:]
            rsi = talib.RSI(subdata[:, 3])[14:]
            obv = talib.OBV(subdata[:, 3], subdata[:, 4])[14:]
            subdata = pd.DataFrame(subdata)
            rolling_mean = subdata.iloc[:, 3].rolling(window=30).mean()
            newdata = subdata.iloc[14:, -2:].assign(stoch=stoch, rsi=rsi, obv=obv, rolling=rolling_mean)
            newdata = newdata.iloc[-1, :].values
            prediction = clf.predict(newdata.reshape((1, -1)))


            subdata = subdata.assign(ave=lambda x: np.mean(x.iloc[:, :4], axis=1))
            myDif, myDea, myBAr = talib.MACD(subdata.ave.values, fastperiod=fast_span, slowperiod=slow_span,
                                          signalperiod=signal_span)


            curr_ave = subdata.iloc[-1, -1]
            lot_value = curr_ave * info.unit_per_lot[ind] * info.margin_rate[ind]
            num_lot = np.round(weight * cost_rate * init_cash / (lot_value * (1. + transaction)))


            flag_sto1 = (stoch[-1] <= 20).astype(int)
            flag_sto2 = (stoch[-1] >= 80).astype(int)
            flag_pred1 = (prediction == 1.).astype(int)
            flag_pred2 = (prediction == 0.).astype(int)
            flag_macd1 = (myDif[-1] - myDif[-2] > 0 and myDif[-1] > myDea[-1]).astype(int)
            flag_macd2 = (myDif[-1] - myDif[-2] < 0 and myDif[-1] < myDea[-1]).astype(int)

            flag_pos = (flag_sto1 + flag_pred1 + flag_macd1)[0]
            flag_neg = (flag_sto2 + flag_pred2 + flag_macd2)[0]


            if flag_pos >= 2:
                if detail_last_min[1] > my_cash_balance_lower_limit:
                    print('----------------Buy', num_lot)
                    position_new[ind] += num_lot

            elif flag_neg >= 2:
                if detail_last_min[1] > my_cash_balance_lower_limit:
                    print('----------------Sell', num_lot)
                    position_new[ind] -= num_lot


    else:
        memory.data_list.append(data)
    return position_new, memory



if __name__ == '__main__':
    pass