# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# クラスの定義始まり
class linearRegression():
    #------------------------------------
    # 1) 学習データおよびモデルパラメータの初期化
    # x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # y: 学習出力データ（データ数のnumpy.array）
    # kernelType: カーネルの種類（文字列：gaussian）
    # kernelParam: カーネルのハイパーパラメータ（スカラー）
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        # 学習データの設定
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]
        
        # カーネルの設定
        self.kernelType = kernelType
        self.kernelParam = kernelParam
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分母の計算にFor文を用いた場合）
    def train(self):
        self.w = np.zeros([self.xDim,1])
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化（行列演算により高速化）
    def trainMat(self):
        self.w = np.zeros([self.xDim,1])
        Xprime = np.vstack([self.x, np.ones(self.y.shape, dtype = np.int)])
        XprimeT = Xprime.T
        inv_sum_xx = np.linalg.inv(np.matmul(Xprime, XprimeT))
        yx = np.matrix(np.matmul(self.y, XprimeT)).T
        return np.matmul(inv_sum_xx, yx)
    #------------------------------------
    
    #------------------------------------
    # 3) 予測
    # x: 入力データ（入力次元 x データ数）
    def predict(self,x):
        y = np.matmul(self.trainMat().T, np.vstack([x, np.ones(x.shape[1], dtype = np.int)]))
        return np.array(y)[0]
    #------------------------------------

    #------------------------------------
    # 4) 二乗損失の計算
    # x: 入力データ（入力次元 x データ数）
    # y: 出力データ（データ数）
    def loss(self,x,y):
        f_x = self.predict(x)
        y = y[np.newaxis]
        loss = np.sum(pow(y - f_x, 2)) / (y - f_x).shape[1]
        return loss
    #------------------------------------
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
    
    # 1) 学習入力次元が2の場合のデーター生成
    myData = rg.artificial(200,100, dataType="1D")
    # myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    # myData = rg.artificial(200,100, dataType="2D")
    # myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)
    
    # 2) 線形回帰モデル
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
    
    # 3) 学習（For文版）
    sTime = time.time()
    # regression.trainMat()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
    
    # 4) 学習（行列版）
    sTime = time.time()
    # regression.trainMat()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    # 5) 学習したモデルを用いて予測
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

    # 6) 学習・評価データおよび予測結果をプロット
    predict = regression.predict(myData.xTest)
    myData.plot(predict,isTrainPlot=False)
    
#メインの終わり
#-------------------
	
