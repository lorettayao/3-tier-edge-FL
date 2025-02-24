# import tensorflow as tf
# import tensorflow_federated as tff
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random

# dirname = '/home/wmnlab/Documents/sheng-ru/HO-Prediction/data/single'
dirname = '/home/wmnlab/Desktop/test/v22'
dirlist = os.listdir(dirname)


def ts_array_create(dirname, dir_list, time_seq):
    features = ['LTE_HO', 'MN_HO',  'SCG_RLF',
                'num_of_neis', 'RSRP', 'RSRQ', 'RSRP1', 'RSRQ1', 'nr-RSRP', 'nr-RSRQ', 'nr-RSRP1', 'nr-RSRQ1']
    target = ['RSRQ', 'RSRP1']
    split_time = []
    for i, f in enumerate(tqdm(dir_list)):
        f = os.path.join(dirname, f)
        df = pd.read_csv(f)
        
        # Check if 'Timestamp' column exists
        if 'Timestamp' not in df.columns:
            print(f"Warning: 'Timestamp' column not found in {f}. Skipping this file.")
            continue
        
        # preprocess data with ffill method
        del df['Timestamp']

        X = df[features]
        Y = df[target]

        Xt_list = []

        for j in range(time_seq):
            X_t = X.shift(periods=-j)
            Xt_list.append(X_t)

        X_ts = np.array(Xt_list)
        X_ts = np.transpose(X_ts, (1, 0, 2))
        X_ts = X_ts[:-(time_seq), :, :]
        X_ts = X_ts.reshape(-1, 320)
        # change 320 to 10
        Y = Y.to_numpy()
        Y = [1 if sum(y) > 0 else 0 for y in Y]

        YY = []

        for j in range(time_seq, len(Y)):
            count = 0
            for k in range(j, len(Y)):
                count += 1
                if Y[k] != 0:
                    break
            YY.append(count)

        YY = np.array(YY)

        split_time.append(len(X_ts))

        if i == 0:
            X_final = X_ts
            Y_final = YY
        else:
            X_final = np.concatenate((X_final, X_ts), axis=0)
            Y_final = np.concatenate((Y_final, YY), axis=0)
    split_time = [(sum(split_time[:i]), sum(split_time[:i])+x)
                  for i, x in enumerate(split_time)]

    return X_final, Y_final, split_time





# Parameters
for _ in range(3):
    dirlist_1 = []
    dirlist_2 = []
    for dn in dirlist:
        tmp = random.randint(0, 1)
        if tmp == 0:
            dirlist_1.append(dn)
        else:
            dirlist_2.append(dn)

    train_examples_1, train_labels_1, train_time_1 = ts_array_create(
        dirname, dirlist_1, time_seq=20)
    train_examples_2, train_labels_2, train_time_2 = ts_array_create(
        dirname, dirlist_2, time_seq=20)
    print("Shape of X_final:", X_final.shape)
    print("Shape of Y_final:", Y_final.shape)
    print("Split times:", split_time)

    