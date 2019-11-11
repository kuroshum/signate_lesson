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
	def __init__(self, x, y, kernelType, kernelParam):
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
		train_x = self.x
		train_x = np.insert(train_x, 1, 1, axis=0)
		xs = train_x.shape[0]
		train_x_t = train_x.transpose()
		train_y = self.y
		trainsum1 = np.zeros((xs,xs))
		trainsum2 = np.zeros((xs,xs))
		
		for i in np.arange(self.x.shape[1]):
			la1 = train_x[:,i]
			la2 = train_x_t[i,:]
			#pdb.set_trace()
			la1 = la1[np.newaxis,:]
			la2 = la2[:,np.newaxis]
			ans1 = la1 * la2
			ans2 = train_y[i] * la2
        	#pdb.set_trace()
			trainsum1 = trainsum1 + ans1
			trainsum2 = trainsum2 + ans2
			
		trainsum1_inv = np.linalg.inv(trainsum1)
		self.w = np.matmul(trainsum1_inv,trainsum2)

	# 2) �ŏ����@��p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ�荂�����j
	def trainMat(self):
		self.w = np.zeros([self.xDim,1])
		mat_x = self.x
		mat_x = np.insert(mat_x, 1, 1, axis=0)
		mat_x_t = mat_x.transpose()
		sum1 = np.matmul(mat_x,mat_x_t)
		mat_y = self.y
		mat_y = mat_y[:,np.newaxis]
		#pdb.set_trace()
		sum2 = np.matmul(mat_x,mat_y)
		sum1_inv = np.linalg.inv(sum1)
		self.w = np.matmul(sum1_inv,sum2)
		#pdb.set_trace()

	def predict(self,x):
		y = []
		predict_x = np.insert(x, 1, 1, axis=0)
		predict_w = self.w.transpose()
		pdb.set_trace()
		y = np.matmul(predict_w,predict_x)
		return y[0]
	#------------------------------------

	#------------------------------------
	# 4) ��摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		loss = 0.0
		loss_y = self.predict(x)
		root_y = y
		root_y = root_y[np.newaxis,:]
		#pdb.set_trace()
		loss_ans = root_y - loss_y
		loss_ans = np.power(loss_ans,2)
		loss = np.mean(loss_ans)
		return loss

	def calcDist(self,x,z):
		pdb.set_trace()
		z_t = z.transpose()
		c_x = np.tile(x,(x.shape[0],z.shape[1],1)).transpose()
		c_z = np.tile(z_t,(z.shape[0],1,x.shape[1])).transpose()
		dist = np.power(c_x - c_z,2)
		pdb.set_trace()
		return dist

	def kernel(self,x):
		dist = self.calcDist(self.x,x)
		if self.kernelType == "gaussian":
			#pdb.set_trace()
			K = np.exp(-1 * (dist / (2 * np.power(self.kernelParam,2))))
		else:
			print("---type error---")
		pdb.set_trace()
		return K

	def trainMatkernel(self):
		self.w = np.zeros([self.xDim,1])
		matk_x = self.kernel(self.x)
		matk_x = np.insert(matk_x, 2, 1, axis=0)
		matk_x = matk_x[:,:,0]
		#matk_x = np.delete(matk_x,0,2)
		
		matk_x_t = matk_x.transpose()
		ksum1 = np.matmul(matk_x,matk_x_t)
		#pdb.set_trace()
		klambda = np.eye(ksum1.shape[0])
		pdb.set_trace()
		ksum1 = ksum1 + (0.01 * klambda)
		matk_y = self.y
		matk_y = matk_y[:,np.newaxis]
		#pdb.set_trace()
		ksum2 = np.matmul(matk_x,matk_y)
		ksum1_inv = np.linalg.inv(ksum1)
		#pdb.set_trace()
		self.w = np.matmul(ksum1_inv,ksum2)
		


	#------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
	
	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="2D")
	myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
	
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
	regression.trainMatkernel()
	eTime = time.time()
	print("train with matrix: time={0:.4} sec".format(eTime-sTime))

	# 5) �w�K�������f����p���ė\��
#	print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	# 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
	kernel = regression.kernel(myData.xTest)[:,:,0]
	pdb.set_trace()
	predict = regression.predict(kernel)
	myData.plot(predict)
	
#���C���̏I���
#-------------------