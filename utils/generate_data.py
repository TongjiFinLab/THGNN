import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable

feature_cols = ['open','high','low','close','to','vol']

path1 = "/home/THGNN-main/data/csi300.pkl"
df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
relation = os.listdir('/home/THGNN-main/data/relation/')
relation = sorted(relation)
date_unique=df1['dt'].unique()
stock_trade_data=date_unique.tolist()
stock_trade_data.sort()

df1['dt']=df1['dt'].astype('datetime64')

def fun(relation_dt, start_dt_month, end_dt_month,df1):
    prev_date_num = 20
    adj_all = pd.read_csv('/home/THGNN-main/data/relation/'+relation_dt+'.csv', index_col=0)
    adj_stock_set = list(adj_all.index)
    pos_g = nx.Graph(adj_all > 0.1)
    pos_adj = nx.adjacency_matrix(pos_g).toarray()
    pos_adj = pos_adj - np.diag(np.diag(pos_adj))
    pos_adj = torch.from_numpy(pos_adj).type(torch.float32)
    neg_g = nx.Graph(adj_all < -0.1)
    neg_adj = nx.adjacency_matrix(neg_g)
    neg_adj.data = np.ones(neg_adj.data.shape)
    neg_adj = neg_adj.toarray()
    neg_adj = neg_adj - np.diag(np.diag(neg_adj))
    neg_adj = torch.from_numpy(neg_adj).type(torch.float32)
    print('neg_adj over')
    print(neg_adj.shape)
    dts = stock_trade_data[stock_trade_data.index(start_dt_month):stock_trade_data.index(end_dt_month)+1]
    print(dts)
    for i in tqdm(range(len(dts))):
        end_data=dts[i]
        start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
        df2 = df1.loc[df1['dt'] <= end_data]
        df2 = df2.loc[df2['dt'] >= start_data]
        code = adj_stock_set
        feature_all = []
        mask = []
        labels = []
        day_last_code = []
        for j in range(len(code)):
            df3 = df2.loc[df2['code'] == code[j]]
            y = df3[feature_cols].values
            if y.T.shape[1] == prev_date_num:
                one = []
                feature_all.append(y)
                mask.append(True)
                label = df3.loc[df3['dt'] == end_data]['label'].values
                labels.append(label[0])
                one.append(code[j])
                one.append(end_data)
                day_last_code.append(one)
        feature_all = np.array(feature_all)
        features = torch.from_numpy(feature_all).type(torch.float32)
        mask = [True]*len(labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        result = {'pos_adj': Variable(pos_adj), 'neg_adj': Variable(neg_adj),  'features': Variable(features),
                  'labels': Variable(labels), 'mask': mask}
        with open('/home/THGNN-main/data/data_train_predict/'+end_data+'.pkl', 'wb') as f:
            pickle.dump(result, f)
        df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
        df.to_csv('/home/THGNN-main/data/daily_stock/'+end_data+'.csv', header=True, index=False, encoding='utf_8_sig')
        
#The first parameter and third parameters indicate the last trading day of each month, and the second parameter indicates the first trading day of each month.
# for i in ['2020','2021','2022']:
#     for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
#         stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
#         fun(stock_m[-1], stock_m[0], stock_m[-1], df1)
fun('2022-11-30','2022-11-01','2022-11-30',df1)
fun('2022-12-30','2022-12-01','2022-12-30',df1)