#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
import pandas as pd
from tqdm.notebook import tqdm

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


# # Functions

# In[40]:


def ts_array_create(data_list, time_seq_len, pred_time, features, ffill_cols=[],two_hot_cols=[],merged_cols=[]):

    X_all = []
    Y_all_cls = []
    Y_all_fst = []
    files_record = []
    
    def vecot_to_num(v):
        num = 0.0
        for i, t in enumerate(v):
            if t != 0:
                num = i+t
                break
        return num

    def replace_zero_with_one(value):
        if value == 0:
            return 0
        else:
            return 1


    count = 0
    for file in tqdm(data_list):

        df = pd.read_csv(file)

        # Hard to change to a feature, delete it now.
        del df['Timestamp'], df['PCI'], df['EARFCN'], df['NR-PCI']

        # Two hot column
        for col in two_hot_cols:
            df[col] = df[col].apply(replace_zero_with_one)
            
        df[ffill_cols] = df[ffill_cols].replace(0, pd.NA)
        df[ffill_cols] = df[ffill_cols].ffill()
        for col in ffill_cols:
            if not pd.notna(df[col].iloc[0]):
                df = df[df[col].notna()]
        df.reset_index(drop=True, inplace=True)
        
        X = df[features]
        # Merged columns
        for cols in merged_cols:
            new_column = X[cols[:-1]].max(axis=1)
            col_num = X.columns.get_loc(cols[0])
            X = X.drop(cols[:-1], axis=1)
            X.insert(col_num, cols[-1], new_column)
        
        target = ['RLF_II', 'RLF_III']
        Y = df[target].copy()
        Y['RLF'] = Y.apply(lambda row: max(row['RLF_II'], row['RLF_III']), axis=1)
        Y.drop(columns=target, inplace=True)

        Xt_list = []
        Yt_list = []

        for i in range(time_seq_len):
            X_t = X.shift(periods=-i)
            X_t = X_t.to_numpy()
            Xt_list.append(X_t)

        Xt_list = np.stack(Xt_list, axis=0)
        Xt_list = np.transpose(Xt_list, (1,0,2))
        Xt_list = Xt_list[:-(time_seq_len + pred_time -1), :, :]

        for i in range(time_seq_len, time_seq_len+pred_time):
            Y_t = Y.shift(periods=-i)
            Y_t = Y_t.to_numpy()
            Yt_list.append(Y_t)

        Yt_list = np.stack(Yt_list, axis=0)
        Yt_list = np.transpose(Yt_list, (1,0,2))
        Yt_list = Yt_list[:-(time_seq_len + pred_time -1), :, :]
        Yt_list = np.squeeze(Yt_list)
        if pred_time == 1: 
            Yt_cls = np.where(Yt_list != 0, 1, 0) 
            Yt_fst = Yt_list
        else: 
            Yt_cls = np.where((Yt_list != 0).any(axis=1), 1, 0)
            Yt_fst = np.apply_along_axis(vecot_to_num, axis=1, arr=Yt_list)

        X_all.append(Xt_list)
        Y_all_cls.append(Yt_cls)
        Y_all_fst.append(Yt_fst)
        files_record.append((file, (count, count +len(Yt_cls))))
        count += len(Yt_cls)
        
    X_all = np.concatenate(X_all, axis=0)
    Y_all_cls = np.concatenate(Y_all_cls, axis=0)
    Y_all_fst = np.concatenate(Y_all_fst, axis=0)
    
    return X_all, Y_all_cls, Y_all_fst, files_record


# In[41]:


# performance
def performance(model, dtest, y_test):
    
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    ACC = accuracy_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred_proba)
    AUCPR = average_precision_score(y_test, y_pred_proba)
    P = precision_score(y_test, y_pred)
    R = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {ACC}; AUC: {AUC}; AUCPR: {AUCPR}; P: {P}; R: {R}; F1: {F1}")
    
    return ACC, AUC, AUCPR, P, R, F1


# In[42]:


# Debug Function
def count_rlf(data_list):
    count = 0
    for f in data_list:
        df = pd.read_csv(f)
        for i in range(len(df)):
            if df['RLF_II'].iloc[i] or df['RLF_III'].iloc[i]:
                count += 1
    return count

def find_original_input(ind, file_record, time_seq_len):
    
    for (file, ind_range) in file_record:
        if ind_range[0]<=ind<ind_range[1]:
            target_file = file    
            tar_ind_range = ind_range
            
    df = pd.read_csv(target_file)

    return df[ind-tar_ind_range[0]:ind-tar_ind_range[0]+time_seq_len], target_file


# # Load Data

# In[43]:


# Set seed
seed = 55688


# In[46]:


# Read Data
data_folder = '/home/wmnlab/Desktop/test/v22'
data_list = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]
data_list.remove(os.path.join(data_folder, 'record.csv'))
test_data_list1 = [x for x in data_list if ('2023-11-01' in x and '#02' in x)] # 同一天機捷 
test_data_list2 = [x for x in data_list if '2023-11-02' in x] # 機捷
test_data_list3 = [x for x in data_list if '2023-11-09' in x] # 棕線
test_data_list4 = test_data_list1 + test_data_list2 + test_data_list3
train_data_list = [x for x in data_list if x not in test_data_list1 + test_data_list2 + test_data_list3]

time_seq_len = 10
pred_time = 3

features = ['num_of_neis', 'RSRP','RSRQ','RSRP1','RSRQ1','nr-RSRP','nr-RSRQ','nr-RSRP1','nr-RSRQ1',
            'E-UTRAN-eventA3','eventA5','NR-eventA3','eventB1-NR-r15',
            'LTE_HO','MN_HO','MN_HO_to_eNB','SN_setup','SN_Rel','SN_HO', 
            'RLF_II', 'RLF_III','SCG_RLF']
ffill_cols = ['RSRP1', 'RSRQ1']
two_hot_vec_cols = ['E-UTRAN-eventA3','eventA5','NR-eventA3','eventB1-NR-r15',
            'LTE_HO','MN_HO','MN_HO_to_eNB','SN_setup','SN_Rel','SN_HO','RLF_II','RLF_III','SCG_RLF']
merged_cols = [['LTE_HO', 'MN_HO_to_eNB', 'LTE_HO'], ['RLF_II', 'RLF_III', 'RLF']]

X_train, y_cls_train, y_fst_train, record_train = ts_array_create(train_data_list, time_seq_len, pred_time, features, ffill_cols,two_hot_vec_cols,merged_cols)
X_train_2d = X_train.reshape(X_train.shape[0], -1)

X_test1, y_cls_test1, y_fst_test1, record_test1 = ts_array_create(test_data_list1, time_seq_len, pred_time, features, ffill_cols,two_hot_vec_cols,merged_cols)
X_test1_2d = X_test1.reshape(X_test1.shape[0], -1)

X_test2, y_cls_test2, y_fst_test2, record_test2 = ts_array_create(test_data_list2, time_seq_len, pred_time, features, ffill_cols,two_hot_vec_cols,merged_cols)
X_test2_2d = X_test2.reshape(X_test2.shape[0], -1)

X_test3, y_cls_test3, y_fst_test3, record_test3 = ts_array_create(test_data_list3, time_seq_len, pred_time, features, ffill_cols,two_hot_vec_cols,merged_cols)
X_test3_2d = X_test3.reshape(X_test3.shape[0], -1)

X_test4, y_cls_test4, y_fst_test4, record_test4 = ts_array_create(test_data_list4, time_seq_len, pred_time, features, ffill_cols,two_hot_vec_cols,merged_cols)
X_test4_2d = X_test4.reshape(X_test4.shape[0], -1)


# In[ ]:


X_train[0].shape


# In[ ]:


# Count RLF number
rlf_num_train = count_rlf(train_data_list)
rlf_num_test1 = count_rlf(test_data_list1)
rlf_num_test2 = count_rlf(test_data_list2)
rlf_num_test3 = count_rlf(test_data_list3)
rlf_num_test4 = count_rlf(test_data_list4)
print(f'RLF # in training data: {rlf_num_train}\nRLF # in testing data1: {rlf_num_test1}\nRLF # in testing data2: {rlf_num_test2}\nRLF # in testing data3: {rlf_num_test3}\n')


# In[ ]:


# 將數據轉換為 DMatrix 格式
dtrain = xgb.DMatrix(X_train_2d, label=y_cls_train)
dtest1 = xgb.DMatrix(X_test1_2d, label=y_cls_test1)
dtest2 = xgb.DMatrix(X_test2_2d, label=y_cls_test2)
dtest3 = xgb.DMatrix(X_test3_2d, label=y_cls_test3)
dtest4 = xgb.DMatrix(X_test4_2d, label=y_cls_test4)


# # Train

# In[ ]:


# xgb parameters
params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'error',  
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'alpha': 0.01,
    'lambda':1.0,
    'seed': seed,
    'tree_method': 'hist',
    'device': 'cuda:0'
}


# In[ ]:


# Model Create
num_rounds = 1000
watchlist = [(dtrain, 'train'), (dtest1, 'test1'), (dtest2, 'test2'), (dtest3, 'test3')] 
model = xgb.train(params, dtrain, num_rounds, evals=watchlist, early_stopping_rounds=20,  verbose_eval=True)


# In[ ]:


# Metric calculate
performance(model, dtrain, y_cls_train)
performance(model, dtest1, y_cls_test1)
performance(model, dtest2, y_cls_test2)
performance(model, dtest3, y_cls_test3)
performance(model, dtest4, y_cls_test4)
pass


# In[ ]:


# save model
save_path = '../model/rlf_cls_xgb.json'
config = model.save_model(save_path)

# how to load
# model2 = xgb.Booster()
# model2.load_model(save_path)


# # Tuning

# In[ ]:


# xgb parameters
params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'error',  
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8, # on data
    'colsample_bytree': 0.8, # on feature
    'lambda': 0.01, # L2
    'alpha': 0.01, # L1
    'seed': seed,
    'tree_method': 'hist',
    'device': 'cuda:0'
}

# Model Create
num_rounds = 200
# watchlist = [(dtrain, 'train')]
# watchlist = [(dtrain, 'train'), (dtest4, 'test4')] 
watchlist = [(dtrain, 'train'), (dtest1, 'test1'), (dtest2, 'test2'), (dtest3, 'test3')] 
model = xgb.train(params, dtrain, num_rounds, evals=watchlist,  early_stopping_rounds=20, verbose_eval=True)

# Metric calculate
performance(model, dtrain, y_cls_train)
performance(model, dtest1, y_cls_test1)
performance(model, dtest2, y_cls_test2)
performance(model, dtest3, y_cls_test3)
performance(model, dtest4, y_cls_test4)
pass


# ## Grid Search

# In[ ]:


from itertools import product

watchlist = [(dtrain, 'train'), (dtest1, 'test1'), (dtest2, 'test2'), (dtest3, 'test3')] 
    
# Define parameters range
learning_rate_values = [0.05, 0.1, 0.2]
max_depth_values = [4, 5, 6, 7, 8]
subsample_values = [0.8, 0.9, 1.0]
colsample_bytree_values = [0.8, 0.9, 1.0]
alphas = [0.01,0.1,1]
lambdas = [0.01,0.1,1]
num_rounds_values = [50,100,200,300]
r = product(learning_rate_values, max_depth_values, subsample_values, colsample_bytree_values, alphas, lambdas)

savefile = '../info/xgb_record_no_early_stopping.csv'
with open(savefile, 'w') as f:
    print('lr, max_d, s_sample, cols_bytree, alpha, lambda, n,ACC(train), AUC(train), AUCPR(train), P(train), R(train), F1(train), ACC(train), AUC(test), AUCPR(test), P(test), R(test), F1(test)',file=f)
    for lr, d, s, cbt, a, l in tqdm(r):
        params = {'objective': 'binary:logistic', 'eval_metric': 'error',  'seed': seed,'tree_method': 'hist','device': 'cuda:0'}
        params['learning_rate'] = lr
        params['max_depth'] = d
        params['subsample'] = s
        params['colsample_bytree'] = cbt
        params['alpha'] = a
        params['lambda'] = l
        for num_rounds in num_rounds_values:
            model = xgb.train(params, dtrain, num_rounds, evals=watchlist,  early_stopping_rounds=20, verbose_eval=False)
            
            record = [lr, d, s, cbt, a, l, num_rounds]
            record+=list(performance(model, dtrain, y_cls_train))
            performance(model, dtest1, y_cls_test1)
            performance(model, dtest2, y_cls_test2)
            performance(model, dtest3, y_cls_test3)
            record += list(performance(model, dtest4, y_cls_test4))
            
            params.clear()
            record = [str(x) for x in record]
            
            print(','.join(record),end='\n', file=f)
            


# In[ ]:





# In[ ]:


params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'error',  
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8, # on data
    'colsample_bytree': 0.8, # on feature
    'lambda': 0, # L2
    'alpha': 0, # L1
    'seed': seed,
    'tree_method': 'hist',
    'device': 'cuda:0'
}


# In[ ]:





# In[ ]:





# In[ ]:


params.clear()


# # Debugging

# In[ ]:


# Spefify 
dtest = dtest4
y_cls, y_fst, X_test = y_cls_test4, y_fst_test4, X_test4

y_pred_proba = model.predict(dtest) 
y_pred = (y_pred_proba > 0.5).astype(int)

FP_input = []
FP_input_ind = []
FP_time = []

FN_input = []
FN_input_ind = []

TP_input = []
TP_input_ind = []
TP_time = []

for i, (pred, label, t, x) in enumerate(zip(y_pred, y_cls, y_fst, X_test)):
    if pred != label:
        if label == 1: # FP analysis
            FP_time.append(t)
            x = pd.DataFrame(x, columns=features)
            FP_input.append(x)
            FP_input_ind.append(i)
        else: # FN analysis
            x = pd.DataFrame(x, columns=features)
            FN_input.append(x)
            FN_input_ind.append(i)
    else: 
        if label == 1: # TP analysis
            TP_time.append(t)
            x = pd.DataFrame(x, columns=features)
            TP_input.append(x)
            TP_input_ind.append(i)
            
len(TP_input), len(FP_input), len(FN_input)


# In[ ]:


# Excel filename
file_record = record_test4
excel_file = '../info/TP_data.xlsx'
input_ind = TP_input_ind

# ExcelWriter 
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    for ind in tqdm(input_ind):
        # x = X_test[ind] 
        # df = pd.DataFrame(x, columns=features)
        df, tar_file = find_original_input(ind, file_record, time_seq_len)
        df['filename'] = [tar_file] + [None]*(len(df) - 1)
        df.to_excel(writer, sheet_name=f'{ind}', index=False)


# In[ ]:


df, _ = find_original_input(316, file_record, time_seq_len)


# In[ ]:


from pprint import pprint
pprint(TP_input_ind[:20])
pprint(TP_time[:20])


# In[ ]:


x = X_test4_2d[621]
x = np.expand_dims(x, axis=0)
X = xgb.DMatrix(x)
model.predict(X)


# In[ ]:


len(FN_input)/1490


# In[ ]:


# Failed CDF
sorted_data = np.sort(FP_time)
cumulative_distribution = np.arange(1, len(sorted_data) + 1) / (len(sorted_data)+len(FN_input))

plt.ylim([0,1])
plt.plot(sorted_data, cumulative_distribution, marker='o', linestyle='-', color='b')
plt.xlabel('Time away from RLF (second)')
# plt.ylabel('Cumulative Distribution Function (CDF)')
plt.title('CDF of the false prediced Data')
plt.grid(True)
plt.show()


# In[ ]:


# feature importance
importance = model.get_score(importance_type='gain')

sorted_importance = {}
for k, v in importance.items():
    num = int(k[1:])
    feature_name = features[num%len(features)]
    sorted_importance[f'{feature_name} {time_seq_len-num//34}'] = v

sorted_importance = sorted(sorted_importance.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


# Plot Feature Importance
top_features = 20

data, labels = [], []
for f, score in reversed(sorted_importance[:top_features]):
    data.append(round(score))
    labels.append(f)

bars = plt.barh(labels, data)
plt.bar_label(bars)

# 設置標題和標籤
plt.title('Feature importance')
plt.xlabel('F score (gain)')
plt.ylabel('Features')

plt.grid()

# 顯示圖表
plt.show()


# In[ ]:




