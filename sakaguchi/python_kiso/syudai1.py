import numpy as np
import pdb

W = np.array([ [1,0,0], [0,1/2,0], [0,0,1/3] ])
H = np.array([ [1,0,0], [0,2,0], [0,0,3] ])
print(f"W:\n{W}\nH:\n{H}\n")


x = np.array([ [1], [2], [3] ])
b = np.array([ [1], [2], [3] ])
print(f"x:\n{x}\nb:\n{b}\n")


W_inv = np.linalg.inv(W)
print(f"W_inv:\n{W_inv}\n")


W_transpose = W.T
f_x = np.matmul(W_transpose,x) 
f_x = f_x + b
print(f"W_inv_transpose:\n{W_transpose}\n")
print(f"f_x:\n{f_x}\n")

e = np.exp(f_x)
print(e)

u = np.sum(e)
print(u)

out = e/u
print(f"out:\n{out}\n")
print(np.sum(out))
