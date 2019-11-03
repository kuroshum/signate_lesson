import regressionData as rg
import numpy as np
import pdb

#regressionDataクラスの初期化
#regressionData(学習データ数,評価データ数,データ種類)

def w_indicate(d):
    myData = rg.artificial(200,100, dataType=d)

    x = myData.xTrain
    
    x = np.insert(x, 1, 1, axis=0)
    xs = myData.xTrain.shape[0]
    x_t = x.transpose()
    y = myData.yTrain
    sum1 = np.zeros((xs,xs))
    sum2 = np.zeros((xs,xs))
    #pdb.set_trace()
    if d == "1D":
        
        for i in np.arange(myData.xTrain.shape[1]):
            la1 = x[:,i]
            la2 = x_t[i,:]
            la1 = la1[np.newaxis,:]
            la2 = la2[:,np.newaxis]
            ans1 = la1 * la2
            ans2 = y[i]*la1
            #pdb.set_trace()
            sum1 = sum1 + ans1
            sum2 = sum2 + ans2
    else:
        for i in np.arange(myData.xTrain.shape[1]):
            sum1 = np.append(sum1,x[:,i]*x_t[i,:])
            sum2 = np.append(sum2,y[i]*x[:,i])
    
    pdb.set_trace()
    sum1_inv = np.linalg.inv(sum1)

    w = sum1_inv * sum2
    #pdb.set_trace()
    print(w)

if __name__ == "__main__":
    w_indicate("1D")


        

