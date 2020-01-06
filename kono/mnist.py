import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_mnist_data():
	path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.pkl")
	if not os.path.exists(path):
		mnist = fetch_openml('mnist_784', version=1)
		with open(path, 'wb') as f:
			pickle.dump(mnist, f, -1)
	with open(path, 'rb') as f:
		mnist = pickle.load(f)
	xData = mnist.data.astype(np.float32)
	xData /= 255
	yData = mnist.target.astype(np.int32)
	return xData, yData

def weight_variable(name, shape):
	return tf.compat.v1.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name, shape):
	return tf.compat.v1.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

def linear_regression(x_t, xDim, yDim, reuse=False):
	with tf.compat.v1.variable_scope('linear_regression') as scope:
		if reuse:
			scope.reuse_variables()
		w = weight_variable('w', [xDim, yDim])
		b = bias_variable('b', [yDim])
		return tf.nn.softmax(tf.add(b, tf.matmul(x_t, w)))

if __name__ == "__main__":
	xData, yData = load_mnist_data()
	yData = np.eye(10)[yData]
	xData_train, xData_test, yData_train, yData_test =  train_test_split(xData, yData, test_size=0.2, random_state=42)
	x_t = tf.compat.v1.placeholder(tf.float32, [None, xData.shape[1]])
	y_t = tf.compat.v1.placeholder(tf.float32, [None, yData.shape[1]])
	learning_rate = tf.constant(0.01, dtype=tf.float32)
	output_train = linear_regression(x_t, x_t.shape[1], y_t.shape[1])
	output_test = linear_regression(x_t, x_t.shape[1], y_t.shape[1], reuse=True)
	loss_square_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_t, logits=output_train))
	loss_square_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_t, logits=output_test))
	opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
	training_step = opt.minimize(loss_square_train)
	sess = tf.compat.v1.Session()
	init = tf.compat.v1.global_variables_initializer()
	sess.run(init)
	loss_train_list = []
	loss_test_list = []
	BATCH_SIZE = 64
	for ite in range(100):
		perm = np.random.permutation(xData_train.shape[0])
		for i in range(0, xData_train.shape[0], BATCH_SIZE):
			batch_X = xData_train[perm[i : i + BATCH_SIZE]]
			batch_Y = yData_train[perm[i : i + BATCH_SIZE]]
			sess.run(training_step, feed_dict={x_t: batch_X, y_t: batch_Y})
		if ite % 10 == 0:
			perm = np.random.permutation(xData_train.shape[0])
			batch_X = xData_train[perm[0 : BATCH_SIZE]]
			batch_Y = yData_train[perm[0 : BATCH_SIZE]]
			loss_train = sess.run(loss_square_train, feed_dict={x_t: batch_X, y_t: batch_Y})
			loss_train_list.append(loss_train)
			perm = np.random.permutation(xData_test.shape[0])
			batch_X = xData_test[perm[0 : BATCH_SIZE]]
			batch_Y = yData_test[perm[0 : BATCH_SIZE]]
			loss_test = sess.run(loss_square_test, feed_dict={x_t: batch_X, y_t: batch_Y})
			loss_test_list.append(loss_test)
			print("Training ite:{0} loss_train:{1:02.3f}, loss_test:{2:02.3f}".format(ite, loss_train, loss_test))
	plt.plot(loss_train_list, "o-", color="#0000FF")
	plt.plot(loss_test_list, "o-", color="#FF0000")
	plt.legend(["train", "test"], fontsize=14)
	plt.xlabel("Iteration", fontsize=14)
	plt.ylabel("loss", fontsize=14)
	plt.show()
