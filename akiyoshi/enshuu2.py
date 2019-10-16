import numpy as np

A = np.array([[1,3,2],[-1,0,1],[2,3,0]])
A_inv = np.linalg.inv(A) 
print(f"A:\n{A_inv}")
ans = np.matmul(A,A_inv)
print(f"ans:\n{ans}")