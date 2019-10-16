# -*- coding: utf-8 -*-



# 数値計算用のライブラリnumpyをnpとしてインポート

#演習２
import numpy as np

import pdb
print('演習２\n')
A=np.array([[1,3,2],[-1,0,1],[2,3,0]])

A_inv=np.linalg.inv(A)

f_x=np.matmul(A,A_inv)

print(A_inv,"\n")
print(f_x,'\n')

#宿題１
print('宿題１\n')
W=np.array([[1,0,0],[0,1/2,0],[0,0,1/3]])
x=np.array([[1],[2],[3]])
a=np.array([])
b=np.array([[1],[2],[3]])
c=np.array([[0],[0],[0]])

a=np.dot(W.T,x)+b

for num in range(3):
    c[num]=np.exp(a[num])

e_sum=np.sum(c)

for num in range(3):
    print('p(y=',num,'|x)=',c[num]/e_sum)
