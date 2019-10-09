# -*- Pythonによる機械学習1 演習2 -*-

# 数値計算用のライブラリとしてnumpyをnpとしてインポート
import numpy as np

# 行列Aを定義
A = np.array([[1,3,2],[-1,0,1],[2,3,0]])
print(f"A:\n{A}\n")

# 逆行列を計算
A_inv = np.linalg.inv(A)
print(f"A_inv:\n{A_inv}\n")

# 行列Aと逆行列A_invの積
AA_inv = np.matmul(A,A_inv)
print(f"A * A_inv:\n{AA_inv}\n")
