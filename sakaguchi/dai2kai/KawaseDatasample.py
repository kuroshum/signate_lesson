import numpy as np
import pandas as pd
import pdb
import codecs

fp = codecs.open('nme_R031.27145.20181019131530.01.csv','r','utf-8','ignore')
line = fp.read().split('\n')[2:]
date = []
value = []
rate = []
#データのうち学習に使うものの割合
train_per = 0.8
#データを日付、数値、レートごとにlist化する
for sentence in line:
    data_list = sentence.split(',')
    if len(data_list)>2:
        date.append(data_list[0])
        value.append(float(data_list[1]))
        rate.append(float(data_list[2]))
#データをPandas DataFrameの形に結合、変換
df = pd.DataFrame({'date':date,'value':value,'rate':rate})
#Pandas DataFrameから数値、レートの値を取り出してくる
data = df.loc[:,['value','rate']].values
#学習に使うデータがどのindexまでかを求める
train_ind = int(len(data)*train_per)
#データを学習、テストに分ける
train_data = data[:train_ind]
test_data = data[train_ind:]
