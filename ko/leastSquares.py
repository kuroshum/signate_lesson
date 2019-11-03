import numpy as np
import regressionData as rg

def train(self, dataType):
    X = self.xTrain
    Xprime = np.vstack([X.T, np.ones(dataType, dtype = np.int)]) # Xprime = [x1, x2, ..., xN, 1].T
    XprimeT = Xprime.T
    Y = np.append(self.yTrain, 1)
    inv_sum_xx = np.sum(np.matmul(Xprime, XprimeT), dtype = np.double) ** -1
    yx = np.matmul(Y, Xprime)
    return inv_sum_xx * yx

# データの生成
myData1 = rg.artificial(200,100,dataType="1D")
myData2 = rg.artificial(200,100,dataType="2D")
w1 = train(myData1, 1)
print(f"w1:\n{w1}\n")
w2 = train(myData2, 2)
print(f"w2:\n{w2}\n")
