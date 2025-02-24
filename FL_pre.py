import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random

dirname = '/home/wmnlab/Documents/sheng-ru/HO-Prediction/data/single'
# dirname = 'home/wmnlab/Desktop/test/v22/single'
dirlist = os.listdir(dirname)


def ts_array_create(dirname, dir_list, time_seq):
    features = ['LTE_HO', 'MN_HO', 'eNB_to_ENDC', 'gNB_Rel', 'gNB_HO', 'RLF', 'SCG_RLF',
                'num_of_neis', 'RSRP', 'RSRQ', 'RSRP1', 'RSRQ1', 'nr-RSRP', 'nr-RSRQ', 'nr-RSRP1', 'nr-RSRQ1']
    target = ['LTE_HO', 'MN_HO']
    split_time = []
    for i, f in enumerate(tqdm(dir_list)):
        f = os.path.join(dirname, f)
        df = pd.read_csv(f)

        # preprocess data with ffill method
        del df['Timestamp'], df['lat'], df['long'], df['gpsspeed']

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


def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(320, tf.nn.softmax, input_shape=(320,),
                              kernel_initializer='zeros'),
        tf.keras.layers.Dense(160, tf.nn.softmax,
                              kernel_initializer='zeros'),
        tf.keras.layers.Dense(80, tf.nn.softmax,
                              kernel_initializer='zeros'),
        tf.keras.layers.Dense(1, tf.nn.softmax,
                              kernel_initializer='zeros')

    ])
    # model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(320,)),
    #     tf.keras.layers.GRU(256, return_sequences=True),
    #     tf.keras.layers.SimpleRNN(128),
    #     tf.keras.layers.Dense(1)
    # ])

    return tff.learning.models.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])


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
    # train_examples_1 = train_examples_1.reshape(train_examples_1.shape[0], 1, train_examples_1.shape[1])
    train_dataset_1 = tf.data.Dataset.from_tensor_slices((
        train_examples_1, train_labels_1)).batch(batch_size=12)
    train_dataset_2 = tf.data.Dataset.from_tensor_slices((
        train_examples_2, train_labels_2)).batch(batch_size=12)

    # Pick a subset of client devices to participate in training.
    train_data = [train_dataset_1, train_dataset_2]
    # print(train_data)
    # print(list(train_dataset_1))
    # Wrap a Keras model for use with TFF.

    # Simulate a few rounds of training with the selected client devices.
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
            learning_rate=30.0),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=30.0))
    state = trainer.initialize()
    state, metrics = trainer.next(state, train_data)
    print(metrics['client_work']['train']['loss'])
