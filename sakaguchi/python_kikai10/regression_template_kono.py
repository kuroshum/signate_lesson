# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# �N���X�̒�`�n�܂�
class linearRegression():
	#------------------------------------
	# 1) �w�K�f�[�^����у��f���p�����[�^�̏�����
	# x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
	# y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
	# kernelType: �J�[�l���̎�ށi������Fgaussian�j
	# kernelParam: �J�[�l���̃n�C�p�[�p�����[�^�i�X�J���[�j
	def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
		# �w�K�f�[�^�̐ݒ�
		self.x = x
		self.y = y
		self.xDim = x.shape[0]
		self.dNum = x.shape[1]
		
		# �J�[�l���̐ݒ�
		self.kernelType = kernelType
		self.kernelParam = kernelParam
	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK��
	# �i����̌v�Z��For����p�����ꍇ�j
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
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ�荂�����j
	def trainMat(self):
		x = np.insert(self.x,self.x.shape[0],1,axis=0)
		l = np.linalg.inv(np.dot(x,x.T))
		r = np.dot(x,self.y.T)
		self.w = np.dot(l,r)
		#self.w = np.zeros([self.xDim,1])
	#------------------------------------
	
	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		x1 = np.insert(x,x.shape[0],1,axis=0)
		y = np.dot(self.w.T,x1)
		return y
	#------------------------------------

	#------------------------------------
	# 4) ��摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		preY = self.predict(x)
		loss = np.sum((y-preY)*(y-preY))/len(y)
		return loss
	#------------------------------------


	#------------------------------------
	# 6) 2�̃f�[�^�W���Ԃ̑S�Ă̑g�ݍ��킹�̋����̌v�Z
	# x: �s��i���� x �f�[�^���j
	# z: �s��i���� x �f�[�^���j
	def calcDist(self,x,z):
		#�y�s��x�̃f�[�^�_x1, x2, ..., xN�ƁA�s��z�̃f�[�^�_z1, z2, ..., zM�Ƃ̊Ԃ�MxN�̋������v�Z�z
		#xTile = np.tile(x,(x.shape[0],z.shape[1],1))
		#zTile = np.tile(z.T,(z.shape[0],1,x.shape[1]))
		#dist = abs(xTile.T-zTile.T)
		#return dist
		return np.sqrt(((x[:, :,np.newaxis] - z[:,  np.newaxis,:]) ** 2).sum(axis=0))
	#------------------------------------

	#------------------------------------
	# 5) �J�[�l���̌v�Z
	# x: �J�[�l�����v�Z����Ώۂ̍s��i���� x �f�[�^���j
	def kernel(self,x):
		#�yself.x�̊e�f�[�^�_xi�ƍs��x�̊e�f�[�^�_xj�ƊԂ̃J�[�l���lk(xi,xj)���e�v�f�Ɏ��O�����s����v�Z�z
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

# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	#------------------------------------
	#liner
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="1D")
	#myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
	# 2) ���`��A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain)
	#regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) �w�K�i�s��Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix  : time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	#myData.plot(predict,isTrainPlot=False)
	myData.plot(predict)



	#------------------------------------
	#2����
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="2D")
	#myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)
	
	# 2) ���`��A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain)
	#regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
		
	# 4) �w�K�i�s��Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix  : time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	#myData.plot(predict,isTrainPlot=False)
	myData.plot(predict)


	#------------------------------------
	#Kernel
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) ���`��A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	# 4) �w�K
	regression.trainMatKernel()
	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)



	#------------------------------------
	#Kernel2
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) ���`��A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=0.1)
	# 4) �w�K
	regression.trainMatKernel()
	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)


	#------------------------------------
	#Kernel3
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	# 2) ���`��A���f��
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=5)
	# 4) �w�K
	regression.trainMatKernel()
	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))
	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict,isTrainPlot=False)

#���C���̏I���
#-------------------
	