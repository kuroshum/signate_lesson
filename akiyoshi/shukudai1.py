import numpy as np
import pdb

W = np.array([ [1,0,0], [0,1/2,0], [0,0,1/3] ])

x = np.array([ [1], [2], [3] ])
b = np.array([ [1], [2], [3] ])

WT = W.T
WTx = np.matmul(WT,x)

WTx_b = WTx + b

exp_WTx_b = np.exp(WTx_b)

sum_exp_WTx_b = np.sum(exp_WTx_b)

softmax = exp_WTx_b/sum_exp_WTx_b

print(f"softmax:\n{softmax}")