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
	def train(self):
		self.w = np.zeros([self.xDim,1])
	#------------------------------------

	#------------------------------------
	# 2) �ŏ����@��p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ�荂�����j
	def trainMat(self):
		self.w = np.zeros([self.xDim,1])
		Xd = self.x
		data = Xd.shape[0]
		ones = np.ones(Xd.shape[1])
		Xd = np.vstack([Xd,ones])
		XdXdT = np.matmul(Xd,Xd.T)
		inv_XdXdT = np.linalg.inv(XdXdT)
		YT = self.y.reshape(1,200)
		YT = YT.T
		YXd = np.matmul(Xd,YT)
		print(inv_XdXdT.shape)
		print(YXd.shape)
		self.w = np.matmul(inv_XdXdT,YXd)    
		return self.w
	#------------------------------------
	
	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		#pdb.set_trace()
		ones = np.ones(x.shape[1])
		pre_x = np.vstack([x,ones])
		y = []
		y = np.matmul(self.w.T,pre_x)
		print("{}".format(y.shape))

		return y[0]
	#------------------------------------

	#------------------------------------
	# 4) ��摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		pre_y = self.predict(x)
		print("{}".format(pre_y.shape))
		print("{}".format(y.shape))
		loss_y = pre_y - y
		sqrt_loss = np.power(loss_y,2)


		loss = np.mean(sqrt_loss)
		return loss
	#------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	myData = rg.artificial(200,100, dataType="2D")
	#myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
	# 2) ���`��A���f��
	#regression = linearRegression(myData.xTrain,myData.yTrain)
	regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
	
	# 3) �w�K�iFor���Łj
	sTime = time.time()
	regression.train()
	eTime = time.time()
	print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
	
	# 4) �w�K�i�s��Łj
	sTime = time.time()
	regression.trainMat()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f����p���ė\��
	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	predict = regression.predict(myData.xTest)
	myData.plot(predict,isTrainPlot=False)
	
#���C���̏I���

	