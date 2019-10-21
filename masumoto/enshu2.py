import numpy as np

# arrayA
A = np.array([ [1,3,2], [-1,0,1], [2,3,0] ])

# inverseA
A_inv = np.linalg.inv(A)
print(f"A_inv:\n{A_inv}\n")

# multiplication
ans = np.matmul(A,A_inv)
print(f"ans:\n{ans}\n")
