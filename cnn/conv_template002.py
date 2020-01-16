'''
###------このプログラムは書きかけです------###
畳み込み層の記述を行った後に全結合にする部分で詰まったので、そこまで書いたところでgitにpushしています。
'''

import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim

class cnn:
    def __init__(self, x, y):
        #placeholder
        self.x = tf.reshape(x, [None, 28, 28, 1])
        self.y = slim.one_hot_encoding(y, 10)

    def cnn_process(self):
        '''
        #------conv1------#
        '''
        # 畳み込み層の次元数
        conv1_dim = 20
        # 重みの生成
        conv1_Filter = tf.Variable(tf.random_normal([5, 5, 1, conv1_dim], stddev=0.1), dtype=tf.float32)
        # バイアス
        bias1 = tf.Variable(tf.random_normal([conv1_dim]))
        # 畳み込みの関数 tf.nn.conv2dを使用
        # 引数1 : input_data
        # 引数2 : 重み
        # 引数3 : ストライド（最初と最後は1固定らしい）
        # 引数4 : ゼロパディングをするかしないか　用語がわからなかったので今回はとりあえず「する」を選択した
        conv1_2d = tf.nn.conv2d(ph_x, conv1_Filter, strides=[1, 1, 1, 1], padding='SAME')
        # バイアスの加算
        conv1_b_a = tf.nn.bias_add(conv1_2d, bias1)
        # RELU関数
        conv1_fin = tf.nn.relu(conv1_b_a)
        # プーリング
        pool_size1 = 2 # プーリングサイズ
        conv1_pool = tf.nn.max_pool(conv1_fin, [1, pool_size1, pool_size1, 1], strides=[1, 1, 1, 1], padding='SAME')

        '''
        #------conv2------#
        '''
        # 畳み込み層の次元数
        conv2_dim = 20
        conv2_Filter = tf.Variable(tf.random_normal([5, 5, conv1_dim, conv2_dim], stddev=0.1), dtype=tf.float32)
        bias2 = tf.Variable(tf.random_normal([conv2_dim]))
        conv2_2d = tf.nn.conv2d(conv1_pool, conv2_Filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2_b_a = tf.nn.bias_add(conv2_2d, bias2)
        conv2_fin = tf.nn.relu(conv2_b_a)
        pool_size2 = 2
        conv2_pool = tf.nn.max_pool(conv2_fin, [1, pool_size2, pool_size2, 1], strides=[1, 1, 1, 1], padding='SAME')

        #pdb.set_trace()
        '''
        #------fully connected 1------#
        '''
        # 全結合層1
        # 1次元データに圧縮
        fullcon_flat = slim.flatten(conv2_pool)
        # 全結合層1の次元数
        fullcon1_dim = 512
        #全結合層1の処理
        fullcon1 = slim.fully_connected(fullcon_flat, fullcon_dim)

        '''
        #------fully connected 2------#
        '''
        # 全結合層2

        # 全結合層2の次元数
        fullcon2_dim = 10
        # 全結合層2の処理
        fullcon2 = slim.fully_connected(fullcon1, fullcon_dim)
        # 代入してるだけ
        conv_last = fullcon2

    '''
    #------loss------#
    '''
    def loss(self,y):
        
        # 交差エントロピー
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, y)
        # 損失関数
        return losses = tf.reduce_mean(cross_entropy)

    '''
    #------accuracy------#
    '''
    def accuracy(self,y)
    # 正解率の計算
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        























