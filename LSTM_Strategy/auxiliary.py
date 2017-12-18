
import numpy as np
import pandas as pd
import h5py
import os


os.chdir('/Users/wangchengming/Documents/HKUST/5013/MSBD5013/pythonplatform/')
scaler = pd.read_json('scaler.json')
min_val = scaler['min'].values
range_val = scaler.range.values


def ave(arr, i, step=45):
    li = []
    for j in range(step):
        li.append(arr[i+j])
    return np.mean(li)


def scaling(series):
    return (series - min_val) / range_val

def unscaling(series):
    return range_val * series + min_val


# Read in h5 data as dict and pandas dataframe
def read_h5(path):
    import pandas as pd
    import h5py

    dicts = {}
    indexes = ['A.DCE', 'AG.SHF', 'AU.SHF', 'I.DCE', 'IC.CFE', 'IF.CFE',
               'IH.CFE', 'J.DCE', 'JM.DCE', 'M.DCE', 'RB.SHF', 'Y.DCE', 'ZC.CZC']
    cols = ['open', 'high', 'low', 'close', 'volumn']

    if 'format1' in path:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
        for i in range(len(keys)):
            dat = pd.read_hdf(path, key=keys[i])
            dicts[keys[i]] = dat

    else:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            for i in range(len(keys)):
                dat = pd.DataFrame(f[keys[i]][:])
                dat.index = indexes
                dat.columns = cols
                dicts[keys[i]] = dat
    return dicts

