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
        Xprime = np.vstack([self.x, np.ones(self.y.shape, dtype = np.int)])
        XprimeT = Xprime.T
        inv_sum_xx = np.linalg.inv(np.matmul(Xprime, XprimeT))
        yx = np.matrix(np.matmul(self.y, XprimeT)).T
        return np.matmul(inv_sum_xx, yx)
    #------------------------------------
    
    #------------------------------------
    # 3) �\��
    # x: ���̓f�[�^�i���͎��� x �f�[�^���j
    def predict(self,x):
        y = np.matmul(self.trainMat().T, np.vstack([x, np.ones(x.shape[1], dtype = np.int)]))
        return np.array(y)[0]
    #------------------------------------

    #------------------------------------
    # 4) ��摹���̌v�Z
    # x: ���̓f�[�^�i���͎��� x �f�[�^���j
    # y: �o�̓f�[�^�i�f�[�^���j
    def loss(self,x,y):
        f_x = self.predict(x)
        y = y[np.newaxis]
        loss = np.sum(pow(y - f_x, 2)) / (y - f_x).shape[1]
        return loss
    #------------------------------------
# �N���X�̒�`�I���
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":
    
    # 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
    myData = rg.artificial(200,100, dataType="1D")
    # myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    # myData = rg.artificial(200,100, dataType="2D")
    # myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)
    
    # 2) ���`��A���f��
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
    
    # 3) �w�K�iFor���Łj
    sTime = time.time()
    # regression.trainMat()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
    
    # 4) �w�K�i�s��Łj
    sTime = time.time()
    # regression.trainMat()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    # 5) �w�K�������f����p���ė\��
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

    # 6) �w�K�E�]���f�[�^����ї\�����ʂ��v���b�g
    predict = regression.predict(myData.xTest)
    myData.plot(predict,isTrainPlot=False)
    
#���C���̏I���
#-------------------
	