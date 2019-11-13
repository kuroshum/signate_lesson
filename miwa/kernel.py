# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

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
		one=np.ones((1,200))
		self.x=np.vstack((self.x,one))
		#$self.x=self.x.reshape(2,200)
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
		x=np.vstack((x,one))
		y = np.matmul(self.w.T,x)
		return y
	#------------------------------------

	#------------------------------------
	# 4) ���摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		loss = 0.0
		#print('x.shape',x.shape)
		#print('w.shape',self.w.shape)
		print(y.shape)
		one=np.ones((100,200))
		x=np.vstack((x,one))
		#print(np.matmul(self.w.T,x).shape)
		loss=y-np.matmul(self.w.T,x)
		#pdb.set_trace()
		loss=np.mean(np.square(loss))
		#pdb.set_trace()
		return loss

	def calcDist(self,x,z):
		#pdb.set_trace()
		#print('x.shape',x.shape)
		#print('z.T.shape',z.T.shape)
		x_2=np.tile(x,(x.shape[0],z.shape[1],1))
		z_2=np.tile(z.T,(x.shape[1],z.shape[0],1))
		#print('x_2',x_2.shape)
		#print('x_2.T',x_2.T.shape)
		#print('z_2',z_2.shape)
		dist=np.abs(x_2.T-z_2)

		#print(dist)
		return dist

	def kernel(self,x):
		#pdb.set_trace()
		K=np.exp(-regression.calcDist(x,self.x)/(2*self.kernelParam**2))
		K=np.squeeze(K)
		#print(K.shape)
		return K

	def trainMatKernel(self):
		K=self.kernel(self.x)
		self.w=np.linalg.inv(np.dot(K,K.T))*(np.dot(self.y,K))
		#print(self.w.shape)
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":


	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="2D")
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)

	# 2) ���`���A���f��
	#regression = linearRegression(myData.xTrain,myData.yTrain)
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)

	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

	# 4) �w�K�i�s���Łj
	sTime = time.time()
	regression.trainMatKernel()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f�����p���ė\��
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))

	# 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)




#���C���̏I����
#-------------------
