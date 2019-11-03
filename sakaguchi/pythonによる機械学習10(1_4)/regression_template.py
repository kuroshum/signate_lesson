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
#	def train(self):
#		self.w = np.zeros([self.xDim,1])
	#------------------------------------

	#------------------------------------
	def train(self):
		x = np.insert(self.x,self.x.shape[0],1,axis=0)
		l = np.linalg.inv(np.dot(x,x.T))
		r = np.dot(x,self.y.T)
		self.w = np.dot(l,r)
	#------------------------------------


	#------------------------------------
	# 2) 最小二乗法を用いてモデルパラメータを最適化（行列演算により高速化）
	def trainMat(self):
		x = np.insert(self.x,self.x.shape[0],1,axis=0)
		l = np.linalg.inv(np.dot(x,x.T))
		r = np.dot(x,self.y.T)
		self.w = np.dot(l,r)
		#self.w = np.zeros([self.xDim,1])
	#------------------------------------
	
	#------------------------------------
	# 3) 予測
	# x: 入力データ（入力次元 x データ数）
	def predict(self,x):
		x1 = np.insert(x,x.shape[0],1,axis=0)
		y = np.dot(self.w.T,x1)
		return y
	#------------------------------------

	#------------------------------------
	# 4) 二乗損失の計算
	# x: 入力データ（入力次元 x データ数）
	# y: 出力データ（データ数）
	def loss(self,x,y):
		preY = self.predict(x)
		loss = np.sum((y-preY)*(y-preY))/len(y)
		return loss
	#------------------------------------
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	
	# 1) 学習入力次元が2の場合のデーター生成
	myData = rg.artificial(200,100, dataType="1D")
	#myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
	# 2) 線形回帰モデル
	regression = linearRegression(myData.xTrain,myData.yTrain)
	#regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	# 3) 学習（For文版）
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) 学習（行列版）
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix  : time={0:.4} sec".format(eTime-sTime))

	# 5) 学習したモデルを用いて予測
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) 学習・評価データおよび予測結果をプロット
	predict = regression.predict(myData.xTest)
	#myData.plot(predict,isTrainPlot=False)
	myData.plot(predict)



	#------------------------------------
	#2次元
	# 1) 学習入力次元が2の場合のデーター生成
	myData = rg.artificial(200,100, dataType="2D")
	#myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)
	
	# 2) 線形回帰モデル
	regression = linearRegression(myData.xTrain,myData.yTrain)
	#regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	# 3) 学習（For文版）
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) 学習（行列版）
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix  : time={0:.4} sec".format(eTime-sTime))

	# 5) 学習したモデルを用いて予測
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) 学習・評価データおよび予測結果をプロット
	predict = regression.predict(myData.xTest)
	#myData.plot(predict,isTrainPlot=False)
	myData.plot(predict)



#メインの終わり
#-------------------
	