import numpy as np
import regressionData as rg

def train(self):
    Xprime = np.vstack([self.xTrain, np.ones(self.yTrain.shape, dtype = np.int)])
    XprimeT = Xprime.T
    inv_sum_xx = np.linalg.inv(np.matmul(Xprime, XprimeT))
    yx = np.matrix(np.matmul(self.yTrain, XprimeT)).T
    return np.matmul(inv_sum_xx, yx)

# データの生成
myData1 = rg.artificial(200,100,dataType="1D")
myData2 = rg.artificial(200,100,dataType="2D")
w1 = train(myData1)
print(f"w1:\n{w1}\n") # 一行目：W1 二行目：b
w2 = train(myData2)
print(f"w2:\n{w2}\n") # 一行目、二行目：W1,W2 三行目：b
