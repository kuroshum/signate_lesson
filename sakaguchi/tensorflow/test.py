import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

print(tf.__version__)
print(np.__version__)
print(pd.__version__)

boston = load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['target'] = boston.target
print(df.head())

X_data = np.array(boston.data)
y_data = np.array(boston.target)
print(X_data[0:1])
print(y_data[0:1])

def norm(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data-mean)/std
X_data = norm(X_data)
print(X_data[0:1])

print(X_data.shape)
ones = np.ones((506,1))
X_data = np.c_[ones,X_data]
print(X_data.shape)

X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=42)
y_train = y_train.reshape(404,1)
y_test = y_test.reshape(102,1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

learning_rate = 0.01
training_epochs = 100
n_dim = X_data.shape[1]
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim,1]))
b = tf.Variable(0.0)

y = tf.add(b,tf.matmul(X,W))
cost = tf.reduce_mean(tf.square(y-Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

cost_history = np.array([])
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:X_train,Y:y_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X:X_train,Y:y_train}))
    if epoch %100 == 0:
        W_val = sess.run(W)
        b_val = sess.run(b)

print(cost_history[1])
print(cost_history[50])
#print(cost_history[100])
print(cost_history[99])



pred_test = sess.run(y,feed_dict={X:X_test})

pred = pd.DataFrame({"実際の不動産価格":y_test[:,0],"予測した不動産価格":pred_test[:,0]})
print(pred.head())
