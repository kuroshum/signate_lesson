#既に6時間毎に分かれ、画像データも含まれた、欠損のないデータを想定
#↑前処理を頑張る

import pickle
import pandas as pd
import numpy as np
import datetime

stride = 6#ずらす時間
cycle = 6#時間毎に取り出す
window_size = 48#window_size

fp = open('/Users/Mare/Desktop/satellite_fill1995/satellite_fill1995.pkl', 'rb')
#fp = open('/home/kurora/satellite_window_data/satellite_window_fill1995_48.pkl','rb')
d = pickle.load(fp)

#台風指定(ID)for文
#for i in np.arange(min(d['ID']),3):
for i in np.arange(min(d['ID']),max(d['ID'])+1):#min 1 max 21+1的な 1-21
    df = d[d['ID'] == i]#データのIDがiのもの全て抜き出す
    print('#--------------------------------------------------------')
    print('ID = {}'.format(i))

    #sliding window for文
    for x in np.arange(round(df.shape[0]/4-stride+1)):#stride分を残したインデックス(時間のみを見ている。)
        #48時間分のwindow設定
        df_new = df[df.index[x*4]:df.index[x*4]+datetime.timedelta(hours=window_size)]
        #別のやり方。　if (df_new.index[-1]-df_new.index[0]).total_seconds() != 172800:#48時間ない時
        if (df_new.index[-1]-df_new.index[0]) != datetime.timedelta(days=2):#48時間ない時
            break
        print('window:{}'.format(x))

        #データの成形
        date_w = df_new.index[::4].strftime('%Y-%m-%dT%H:%M:%S')[np.newaxis,:]
        lon_w = df_new[::4]['lon'].to_numpy()[np.newaxis,:]
        lat_w = df_new[::4]['lat'].to_numpy()[np.newaxis,:]
        #画像データのnumpy化
        #typeで分けた方が良い(前処理しっかりしていたら問題ない)
        wide_n = df_new[1::4]['img'].to_numpy()
        ty_n = df_new[0::4]['img'].to_numpy()
        WV_n = df_new[2::4]['img'].to_numpy()
        Diff_ty_n = df_new[3::4]['img'].to_numpy()
        #ラベル
        #true_n:計算用
        #true_w:windowのtrue
        true_n = df_new[::4]['cp'][-1]-df_new[::4]['cp'][0]
        true_w = 1
        if -45 > true_n:
            true_w = 0

        #window内の時間毎のデータ for文(画像データの成形)
        for n in np.arange(0,round(window_size/cycle)+1):#0-8
            #画像について
            #1,241,321,1のみたいに2次元増やしてる。
            #wide_w4とかが1時間の各画像データ
            #wide_wとかが1windowの各画像全体
            wide_n4 = wide_n[n][np.newaxis,:,:,np.newaxis]
            ty_n4 = ty_n[n][np.newaxis,:,:,np.newaxis]
            WV_n4 = WV_n[n][np.newaxis,:,:,np.newaxis]
            Diff_ty_n4 = Diff_ty_n[n][np.newaxis,:,:,np.newaxis]
            if n == 0:#初期化
                wide_w = wide_n4
                ty_w = ty_n4
                WV_w = WV_n4
                Diff_ty_w = Diff_ty_n4
            else:#時間毎の追加処理
                wide_w = np.append(wide_w,wide_n4,axis=3)
                ty_w = np.append(ty_w,ty_n4,axis=3)
                WV_w = np.append(WV_w,WV_n4,axis=3)
                Diff_ty_w = np.append(Diff_ty_w,Diff_ty_n4,axis=3)
        #1windowのデータ成形完了

        #window毎に追加
        if x == 0:#初期化
            date = date_w
            lon = lon_w
            lat = lat_w
            wide = wide_w
            ty = ty_w
            WV = WV_w
            Diff_ty = Diff_ty_w
            true = true_w
        else:#window毎の追加処理
            date = np.append(date,date_w,axis=0)
            lon = np.append(lon,lon_w,axis=0)
            lat = np.append(lat,lat_w,axis=0)
            wide = np.append(wide,wide_w,axis=0)
            ty = np.append(ty,ty_w,axis=0)
            WV = np.append(WV,WV_w,axis=0)
            Diff_ty = np.append(Diff_ty,Diff_ty_w,axis=0)
            true = np.append(true,true_w)
        #1台風のデータ成形完了
        #sliding window 終わり

    print('#-------------------------------')
    print('date.shape   :{}'.format(date.shape))
    print('lon.shape    :{}'.format(lon.shape))
    print('lat.shape    :{}'.format(lat.shape))
    print('wide.shape   :{}'.format(wide.shape))
    print('ty.shape     :{}'.format(ty.shape))
    print('WV.shape     :{}'.format(WV.shape))
    print('Diff_ty.shape:{}'.format(Diff_ty.shape))
    print('true.shape   :{}'.format(true.shape))
    print('NUM          :{}'.format(df['NUM'][0]))
    print('ID           :{}'.format(df['ID'][0]))
    #参考ファイルは(21,11)だが、今回はcolumnsにshipsが入っていないため、(21,10)
    #pandas.Seriesに台風(ID)毎に格納
    if i == min(d['ID']):#初期化
        date_p = pd.Series([date])
        lon_p = pd.Series([lon])
        lat_p = pd.Series([lat])
        wide_p = pd.Series([wide])
        ty_p = pd.Series([ty])
        WV_p = pd.Series([WV])
        Diff_ty_p = pd.Series([Diff_ty])
        true_p = pd.Series([true])
        NUM_p = pd.Series(df['NUM'][0])
        ID_p = pd.Series(df['ID'][0])
    else:#ID毎の追加処理
        i2 = i-min(d['ID'])
        date_p[i2] = date
        lon_p[i2] = lon
        lat_p[i2] = lat
        wide_p[i2] = wide
        ty_p[i2] = ty
        WV_p[i2] = WV
        Diff_ty_p[i2] = Diff_ty
        true_p[i2] = true
        NUM_p[i2] = df['NUM'][0]
        ID_p[i2] = df['ID'][0]

#pandas.DataFrameに成形
out = pd.concat([date_p,lon_p,lat_p,wide_p,ty_p,WV_p,Diff_ty_p,true_p,NUM_p,ID_p],names=['date','lon','lat','wide','ty','WV','Diff_ty','true','NUM','ID',],axis=1)
print(out.shape)
fp.close()

