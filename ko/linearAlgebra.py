# -*- Pythonによる機械学習1 宿題1 -*-

# 数値計算用のライブラリnumpyをnpとしてインポート
import numpy as np
import pdb

# 3*3のnumpy配列（行列）
W = np.array([[1,0,0],[0,1/2,0],[0,0,1/3]])
H = np.array([[1,0,0],[0,2,0],[0,0,3]])
# print(f"W:\n{W}\nH:{H}\n")

# 3*1のnumpy配列（ベクトル）
x = np.array([[1],[2],[3]])
b = np.array([[1],[2],[3]])
# print(f"x:\n{x}\nb:{b}\n")

# 逆行列の計算
W_inv = np.linalg.inv(W)
# print(f"W_inv:\n{W_inv}\n")

# 行列W_invの転置とベクトルxの掛け算とbとの足し算
W_inv_transpose = W_inv.T
f_x = np.matmul(W_inv_transpose,x)
f_x = f_x + b
# print(f"W_inv_transpose:\n{W_inv_transpose}\n")
# print(f"f_x:\n{f_x}\n")

"""
# Wの1行目
row1W = W[[0],:]
# Wの2列目
col2W = W[:,[1]]
print(f"1st row of W:\n{row1W}\n2nd colmn of W:\n{col2W}\n")

# 行ベクトルと列ベクトルの掛け算
rowXcol = np.matmul(row1W,col2W)
print(f"row * col vectors:\n{rowXcol}\n")

# 列ベクトルと行ベクトルの掛け算
colXrow = np.matmul(col2W,row1W)
print(f"col * row vectors:\n{colXrow}\n")

# 行列のアダマール積（要素ごとの積）
g_H = W*H
print("g(H):\n{}\n".format(g_H))
"""

# ソフトマックスを計算する
W_transpose = W.T
f_s = np.matmul(W_transpose,x)
f_s = f_s + b #expの内部
exp_f_s = np.exp(f_s) #分子
# print(f"exp_f_s\n{exp_f_s}\n")
sum_exp_f_s = np.sum(exp_f_s[:]) #分母
# print(f"sum_exp_f_s\n{sum_exp_f_s}\n")
softmax = exp_f_s / sum_exp_f_s
print(f"softmax\n{softmax}\n")
# 合計が1になることを確認
# softmax_sum = np.sum(softmax[:])
# print(f"softmax_sum\n{softmax_sum}\n")
