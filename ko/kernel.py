import numpy as np
import regressionData as rg
import time
import pdb

#---------------------------------

class linearRegression():
    
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]
        self.kernelType = kernelType
        self.kernelParam = kernelParam
    
    def trainMat(self):
        self.w = np.zeros([self.xDim,1])
        Xprime = np.vstack([self.x, np.ones(self.y.shape, dtype = np.int)])
        XprimeT = Xprime.T
        inv_sum_xx = np.linalg.inv(np.matmul(Xprime, XprimeT))
        yx = np.matrix(np.matmul(self.y, XprimeT)).T
        return np.matmul(inv_sum_xx, yx)
    
    def predict(self,x):
        y = np.matmul(self.w, np.concatenate([x, np.ones((1, x.shape[1]))]))
        return y
    
    def loss(self,x,y):
        f_x = self.predict(x)
        y = y[np.newaxis]
        loss = np.sum(pow(y - f_x, 2)) / (y - f_x).shape[1]
        return loss
        
    # 5) カーネルの計算
    # x：カーネルを計算する対象の行列（次元＊データ数）
    def kernel(self, x):
        # self.xの各データ点xiと行列xの各データ点xjと間のカーネル値k（xi、xj）を各要素に持つグラム行列を計算
        # exp(-(||x-x'||^2)/(2*sigma^2))
        return np.exp(-(pow(self.calcDist(x, self.x), 2) / (2 * pow(self.kernelParam, 2))))
    
    # 6) 2つのデータ集合間の全ての組み合わせの距離の計算
    # x：行列（次元＊データ数）
    # z：行列（次元＊データ数）
    def calcDist(self, x, z):
        # 行列xのデータ点N個と行列zのデータ点M個との間のM*N個の距離を計算
        dist = np.linalg.norm(x[:, np.newaxis, :] - z[:, :, np.newaxis], axis = 0)
        return dist
    
    def trainMatKernel(self):
        k = self.kernel(self.x)
        x = np.concatenate([k, np.ones((1, k.shape[1]))])
        self.w = np.matmul(np.linalg.inv(np.matmul(x, x.T) + 0.01 * np.identity(x.shape[0])), np.sum(self.y * x, axis=1))
    
#---------------------------------

if __name__ == "__main__":
    
    # 1) 学習入力次元が2の場合のデーター生成
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    # 2) 線形回帰モデル
    regression1 = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
    regression01 = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=0.1)
    regression5 = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=5)
    # 4) 学習
    regression1.trainMatKernel()
    regression01.trainMatKernel()
    regression5.trainMatKernel()
    # 5) 学習したモデルを用いて予測
    print("loss1={0:.3}".format(regression1.loss(regression1.kernel(myData.xTest), myData.yTest)))
    print("loss0.1={0:.3}".format(regression01.loss(regression01.kernel(myData.xTest), myData.yTest)))
    print("loss5={0:.3}".format(regression5.loss(regression5.kernel(myData.xTest), myData.yTest)))
    # 6) 学習・評価データおよび予測結果をプロット
    # sigma = 1
    kernel = regression1.kernel(myData.xTest)
    predict = regression1.predict(kernel)
    myData.plot(predict,isTrainPlot=False)
    # sigma = 0.1
    kernel = regression01.kernel(myData.xTest)
    predict = regression01.predict(kernel)
    myData.plot(predict,isTrainPlot=False)
    # sigma = 5
    kernel = regression5.kernel(myData.xTest)
    predict = regression5.predict(kernel)
    myData.plot(predict,isTrainPlot=False)
