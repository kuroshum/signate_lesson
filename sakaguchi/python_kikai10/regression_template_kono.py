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


	#------------------------------------
	# 6) 2つのデータ集合間の全ての組み合わせの距離の計算
	# x: 行列（次元 x データ数）
	# z: 行列（次元 x データ数）
	def calcDist(self,x,z):
		#【行列xのデータ点x1, x2, ..., xNと、行列zのデータ点z1, z2, ..., zMとの間のMxN個の距離を計算】
		#xTile = np.tile(x,(x.shape[0],z.shape[1],1))
		#zTile = np.tile(z.T,(z.shape[0],1,x.shape[1]))
		#dist = abs(xTile.T-zTile.T)
		#return dist
		return np.sqrt(((x[:, :,np.newaxis] - z[:,  np.newaxis,:]) ** 2).sum(axis=0))
	#------------------------------------

	#------------------------------------
	# 5) カーネルの計算
	# x: カーネルを計算する対象の行列（次元 x データ数）
	def kernel(self,x):
		#【self.xの各データ点xiと行列xの各データ点xjと間のカーネル値k(xi,xj)を各要素に持つグラム行列を計算】
		ker = self.calcDist(self.x,x)
		K = np.exp(-pow(ker,2)/(2*pow(self.kernelParam,2)))
		return K
		#return np.exp(-(self.calcDist(x, self.x) ** 2 / (2 * self.kernelParam ** 2)))
	#------------------------------------

	#------------------------------------
	def trainMatKernel(self):
		tKer = self.kernel(self.x)
		x = np.insert(tKer,tKer.shape[0],1,axis=0)
		ramda = 0.01
		i = np.eye(x.shape[0])
		l = np.linalg.inv(np.dot(x,x.T)+ramda*i)
		r = np.dot(x,self.y.T)
		self.w = np.dot(l,r)
		#self.w = np.zeros([self.xDim,1])
	#------------------------------------

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	
	#------------------------------------
	#liner
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
	#Kernel
	# 1) 学習入力次元が2の場合のデーター生成
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) 線形回帰モデル
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	# 4) 学習
	regression.trainMatKernel()
	# 5) 学習したモデルを用いて予測
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) 学習・評価データおよび予測結果をプロット
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)



	#------------------------------------
	#Kernel2
	# 1) 学習入力次元が2の場合のデーター生成
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) 線形回帰モデル
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=0.1)
	# 4) 学習
	regression.trainMatKernel()
	# 5) 学習したモデルを用いて予測
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) 学習・評価データおよび予測結果をプロット
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)


	#------------------------------------
	#Kernel3
	# 1) 学習入力次元が2の場合のデーター生成
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) 線形回帰モデル
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=5)
	# 4) 学習
	regression.trainMatKernel()
	# 5) 学習したモデルを用いて予測
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) 学習・評価データおよび予測結果をプロット
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)

#メインの終わり
#-------------------
	