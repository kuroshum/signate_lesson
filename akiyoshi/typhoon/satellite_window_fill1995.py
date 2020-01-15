# -*- coding: utf-8 -*-
import numpy as np
import pdb
import os
import glob
import pandas as pd
import pickle
from datetime import datetime, timedelta

df = pd.DataFrame(columns = ['date','lon','lat','wide','ty','WV','Diff_ty','true','NUM','ID'])

path = '/Users/Mare/Desktop/satellite_fill1995/satellite_fill1995.pkl'
with open(path,'rb') as fp:
    pick = pickle.load(fp)
    print(pick.head())
    print(pick.index[0])
    print(pick.index[8])
    start = 0
    end = 0
    print(pick.shape)

    d_start = pick.index[start]
    d_end = pick.index[end]
    td_limit = timedelta(days=2)
    td_dis = timedelta(hours = 6)
    td = d_end-d_start
    id_max = pick.ID.max()
    print(id_max)
    datanum = 0
    ty_id = 1
    sliding = pd.DataFrame(columns = ['data','lon','lat','wide','ty','WV','Diff_ty','true','NUM','ID'])
    date = [d_start.strftime("%Y-%m-%dT%H:%M:%S")]
    pick_slide = []
    count = 0
    print(date)
    for num in range(1,3):
    #for num in range(1,id_max+1):

        data_ID = pick[pick.ID == num]
        for sl in np.arange(round(data_ID.shape[0]/4-8)):
            win = data_ID[data_ID.index[sl*4]:data_ID.index[sl*4]+td_limit]
            if(win.index[-1]-win.index[0] != td_limit):
                break
            #print(win.index)

            w_date = win.index[::4].strftime('%Y-%m-%dT%H:%M:%S')[np.newaxis,:]
            #print(w_date)
            
            w_lon = win[::4]['lon'].to_numpy()[np.newaxis,:]
            w_lat = win.lat[::4].to_numpy()[np.newaxis,:]

            wide_n = win[1::4]['img'].to_numpy()
            ty_n = win[0::4]['img'].to_numpy()
            WV_n = win[2::4]['img'].to_numpy()
            Diff_ty_n = win[3::4]['img'].to_numpy()
            

            true_n = win[::4]['cp'][-1]-win[::4]['cp'][0]
            true_w = 1
            if -45 > true_n:
                true_w = 0

            #window内の時間毎のデータ for文(画像データの成形)
            for n in np.arange(0,8):
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
            if sl == 0:#初期化
                date = w_date
                lon = w_lon
                lat = w_lat
                wide = wide_w
                ty = ty_w
                WV = WV_w
                Diff_ty = Diff_ty_w
                true = true_w
            else:#window毎の追加処理
                date = np.append(date,w_date,axis=0)
                lon = np.append(lon,w_lon,axis=0)
                lat = np.append(lat,w_lat,axis=0)
                wide = np.append(wide,wide_w,axis=0)
                ty = np.append(ty,ty_w,axis=0)
                WV = np.append(WV,WV_w,axis=0)
                Diff_ty = np.append(Diff_ty,Diff_ty_w,axis=0)
                true = np.append(true,true_w)
            
            #1台風のデータ成形完了
            #sliding window 終わり
        
        pd_date = pd.Series([date])
        pd_lon = pd.Series([lon])
        pd_lat = pd.Series([lat])
        pd_wide = pd.Series([wide])
        pd_ty = pd.Series([ty])
        pd_WV = pd.Series([WV])
        pd_Diff = pd.Series([Diff_ty])
        pd_true = pd.Series([true])
        pd_NUM = pd.Series(pick.NUM[0])
        pd_ID = pd.Series(pick.ID[0])

        #pd_all = np.concatenate(pd_date,pd_lon,pd_lat,pd_wide,pd_ty,pd_WV,pd_Diff,pd_true,pd_NUM,pd_ID,axis = 1)

        np.concatenate([df.date , pd_date],axis = 0)
        if num == 1:
            goal = pd.concat([pd_date,pd_lon,pd_lat,pd_wide,pd_ty,pd_WV,pd_Diff,pd_true,pd_NUM,pd_ID],axis=1)
            print(goal.shape)
        else:
            goal_add = pd.concat([pd_date,pd_lon,pd_lat,pd_wide,pd_ty,pd_WV,pd_Diff,pd_true,pd_NUM,pd_ID],axis=1)
            goal = pd.concat([goal,goal_add],axis = 0)
            print("concat")
           

            print(goal[0])
            
        #df.date = df['date'].append(pd_date)
        
        

        if num == min(pick['ID']):#初期化
            date_p = pd.Series([date])
            lon_p = pd.Series([lon])
            lat_p = pd.Series([lat])
            wide_p = pd.Series([wide])
            ty_p = pd.Series([ty])
            WV_p = pd.Series([WV])
            Diff_ty_p = pd.Series([Diff_ty])
            true_p = pd.Series([true])
            NUM_p = pd.Series(pick['NUM'][0])
            ID_p = pd.Series(pick['ID'][0])
        else:#ID毎の追加処理
            i2 = num-min(pick['ID'])
            date_p[i2] = date
            lon_p[i2] = lon
            lat_p[i2] = lat
            wide_p[i2] = wide
            ty_p[i2] = ty
            WV_p[i2] = WV
            Diff_ty_p[i2] = Diff_ty
            true_p[i2] = true
            NUM_p[i2] = pick['NUM'][0]
            ID_p[i2] = pick['ID'][0]
    #sliding = pd.concat([sliding,goal],axis = 0)
    out = pd.concat([date_p,lon_p,lat_p,wide_p,ty_p,WV_p,Diff_ty_p,true_p,NUM_p,ID_p],names=['date','lon','lat','wide','ty','WV','Diff_ty','true','NUM','ID',],axis=1)
    print(out.columns)
    sliding = pd.DataFrame(goal,columns = ['date','lon','lat','wide','ty','WV','Diff_ty','true','NUM','ID'])
    #print(pd_all.shape)
    print(sliding.date[1])

    
    


    
    

    
    

