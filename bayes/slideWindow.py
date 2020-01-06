import csv
import numpy as np
import pickle
import pandas as pd
import pdb
with open('/Volumes/kesu/lab/satellite_data/satellite_fill1995.pkl', 'rb') as f:
  data_old = pickle.load(f)
  #pdb.set_trace()
  #print(data_old)

slide=6
window=48
t=-45
i=1
true=[]
lon=data_old['lon'][0::4]
lat=data_old['lat'][0::4]
cp=data_old['cp'][0::4]
data_new=pd.concat([lon,lat,cp],axis=1)
wide=data_old['img'][0::4]
data_new=pd.concat([data_new,wide],axis=1)
data_new=data_new.rename(columns={'img':'wide'})
ty=data_old['img'][1::4]
data_new=pd.concat([data_new,ty],axis=1)
data_new=data_new.rename(columns={'img':'ty'})
WV=data_old['img'][2::4]
data_new=pd.concat([data_new,WV],axis=1)
data_new=data_new.rename(columns={'img':'WV'})
Diff_ty=data_old['img'][3::4]
data_new=pd.concat([data_new,Diff_ty],axis=1)
data_new=data_new.rename(columns={'img':'Diff_ty'})
#pdb.set_trace()
j=0
num=data_old['NUM'][1::4]
while j+48<data_new.shape[0]:
    change=data_new['cp'][window+j]-data_new['cp'][j]
    if change <t:
        num[j]=0
    else:
        num[j]=1
    j+=1
while j<data_new.shape[0]:
    num[j]=1
    j+=1
#pdb.set_trace()
data_new=pd.concat([data_new,num],axis=1)
data_new=data_new.rename(columns={'NUM':'true'})
#pdb.set_trace()
num=data_old['NUM'][0::4]
id=data_old['ID'][0::4]
data_new=pd.concat([data_new,num,id],axis=1)
data_new = data_new.reset_index()
pdb.set_trace()
