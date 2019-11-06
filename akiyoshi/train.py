import numpy as np
import regressionData as rg

def train(self):
	#np.matmul
    
	Xd = self.xTrain
	data = Xd.shape[0]
	ones = np.ones(data)
	Xd = np.vstack([Xd.T,ones])
	Y1 = np.append(self.yTrain,0)
	YXd = np.matmul(Y1,Xd)    
	"""
	XdXdT = np.matmul(Xd,Xd.T)
	sum_XdXdT = np.sum(XdXdT) ** -1
	sum_YXd = np.sum(YXd)
	w = np.matmul(sum_XdXdT,sum_YXd)
	"""  
	sum_XdXdT = np.zeros(data)
	sum_YXd = np.zeros(data)
	w = np.zeros(data)        
	#print(Xd.T)
	#print(Y1.shape)
	for i in range(data):
		Xdi = Xd[:,[i]].reshape(1,201)  
		XdiT = Xdi.transpose()              
		#print(XdiT)
		#print(Xd.T[i,:].shape)
		sum_XdXdT[i] = np.sum(np.matmul(XdiT,Xd.T[[i],:])) ** -1
		sum_YXd[i] = np.sum(YXd[i])
		w[i] = sum_XdXdT[i]*sum_YXd[i]     

	return w

myData1 = rg.artificial(200,100,dataType="1D")
myData2 = rg.artificial(200,100,dataType="2D")
w1 = train(myData1)
w2 = train(myData2)
print("w1:\n",w1)
print("w2:\n",w2)