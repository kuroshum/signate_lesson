import numpy as np

A = np.array([ [1,3,2], [-1,0,1], [2,3,0] ])

A_inv = np.linalg.inv(A)
print(f"A:\n{A}\n")
print(f"A_inv:\n{A_inv}\n")

W = np.matmul(A,A_inv)
print(f"A・A_inv:\n{W}\n")
