# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb
from sklearn.metrics import mean_squared_error

#-------------------

class linearRegression():

	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):

		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]

		# �J�[�l���̐ݒ�
		self.kernelType = kernelType
		self.kernelParam = kernelParam

	def train(self):
		self.w = np.zeros([self.xDim,1])

	#------------------------------------
	# 2) �ŏ������@���p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ��荂�����j
	def trainMat(self):
		self.w = np.zeros([self.xDim,1])
	#------------------------------------
		one=np.ones((200,1))
		self.x=np.append(self.x,one)
		self.x=self.x.reshape(2,200)
		x_sum=np.matmul(self.x,self.x.T)
		x_inv=np.linalg.inv(x_sum)
		y_sum=np.matmul(self.y.T,self.x.T)
		self.w=np.matmul(x_inv,y_sum)

	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		y = []
		one=np.ones((1,100))
		x=np.append(x,one)
		x=x.reshape(2,100)
		y = np.matmul(self.w.T,x)
		return y
	#------------------------------------

	#------------------------------------
	# 4) ���摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		loss = 0.0
		one=np.ones((1,100))
		x=np.append(x,one)
		x=x.reshape(2,100)

		loss=y-np.matmul(self.w.T,x)
		#pdb.set_trace()
		loss=np.mean(np.square(loss))
		#pdb.set_trace()

		return loss
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":

	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="1D")
	#myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)

	# 2) ���`���A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain)
	#regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)

	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

	# 4) �w�K�i�s���Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f�����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	myData.plot(predict)

#���C���̏I����
#-------------------
