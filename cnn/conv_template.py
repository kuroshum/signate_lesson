'''
###------このプログラムは書きかけです------###
畳み込み層の記述を行った後に全結合にする部分で詰まったので、そこまで書いたところでgitにpushしています。
'''

import numpy as np
import pandas as pd
import pdb
import tensorflow as tf

#placeholder
ph_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
ph_y = tf.placeholder(tf.float32, shape=(None, 10))

'''
#------conv1------#
'''
# 畳み込み層の次元数
conv1_dim = 20
# 重みの生成
conv1_Filter = tf.Variable(tf.random_normal([5, 5, 1, conv1_dim], stddev=0.1), dtype=tf.float32)
# バイアス
bias1 = tf.Variable(tf.constant(0.1, shape=[conv1_dim]), dtype=tf.float32)
# 畳み込みの関数 tf.nn.conv2dを使用
# 引数1 : input_data
# 引数2 : 重み
# 引数3 : ストライド（最初と最後は1固定らしい）
# 引数4 : ゼロパディングをするかしないか　今回はとりあえずすることとした
conv1_2d = tf.nn.conv2d(ph_x, conv1_Filter, strides=[1, 1, 1, 1], padding='SAME')
# バイアスの加算
conv1_b_a = tf.nn.bias_add(conv1_2d, bias1)
# RELU関数
conv1_fin = tf.nn.relu(conv1_b_a)
# プーリング
pool_size1 = 2 # プーリングサイズ
conv1_pool = tf.nn.max_pool(conv1_fin, [1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], padding='SAME')

'''
#------conv2------#
'''
# 畳み込み層の次元数
conv2_dim = 20
conv2_Filter = tf.Variable(tf.random_normal([5, 5, conv1_dim, conv2_dim], stddev=0.1), dtype=tf.float32)
bias2 = tf.Variable(tf.constant(0.1, shape=[conv2_dim]), dtype=tf.float32)
conv2_2d = tf.nn.conv2d(ph_x, conv2_Filter, strides=[1, 1, 1, 1], padding='SAME')
conv2_b_a = tf.nn.bias_add(conv2_2d, bias2)
conv2_fin = tf.nn.relu(conv2_b_a)
pool_size2 = 2
conv2_pool = tf.nn.max_pool(conv2_fin, [1, pool_size2, pool_size2, 1], [1, pool_size2, pool_size2, 1], padding='SAME')
pdb.set_trace()




