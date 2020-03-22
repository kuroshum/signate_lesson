from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import os
import sys
import pickle
import pdb
from natsort import natsorted
import glob
from datetime import datetime
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn import metrics
from sklearn.model_selection import KFold
#from pre_processing_sample import Data
#import pAUC_method_sample_4_layer as pAUC
FIRST_YEAR = 1995
LAST_YEAR = 2017
window_size = sys.argv[1]
window_data_subpath = '/home/kurora/weatherForecast/typhoon/satellite_window_data/window_size_{}/'.format(window_size)
pkl_subpath = '/home/kurora/weatherForecast/typhoon/pAUC/pickle/window_data'
# ------------------------------------------------------------------------------------------------------------
# リストをバイナリファイルから読み込み
def read_pickle(window_size=48):
    pkl_path = os.path.join(pkl_subpath, 'window_data_{}.pkl'.format(window_size))
    with open(pkl_path,'rb') as fp:
        all_data = pickle.load(fp)
        window_data = pickle.load(fp)
    return all_data, window_data
# ------------------------------------------------------------------------------------------------------------
#-------------------------------------------------
# SHIPS特徴量(時系列データ)を取得
#myData = Data()
#myData.read_pickle()
all_data, window_data = read_pickle(window_size=window_size)
# SHIPS特徴量
#all_data_ships = all_data
# スライディングウィンドウをした後のデータ
window_data_ships = window_data[window_data.index>=datetime(FIRST_YEAR, 7, 1)]
#-------------------------------------------------
#pdb.set_trace()
#-------------------------------------------------
# 閾値の設定
# holdout
holdout_threshold = datetime(2012,12,31)
# センサーの変更によりデータ品質が変化
dev_threshold = datetime(2005,12,31)
#-------------------------------------------------
#-------------------------------------------------
# NN-pAUC のパラメータ設定
# 隠れ層のノード数
h_dims = 24
# レイヤー数
layer = 4
# 入力データの次元
x_dims = 24
keep_prob = 0.5
#-------------------------------------------------
#-------------------------------------------------
# setting of CNN
# CNNのはじめのチャネル数
c_cnn = 32
# カーネルサイズ
kernal_size = 5
# wideデータのチャネル数
c_wide = 9
# tyデータのチャネル数
#c_ty = 27
c_ty = 1
#-------------------------------------------------
#-------------------------------------------------
# setting of image
# wideデータの高さ
height_wide = 321
# wideデータの幅
width_wide = 241
height_wide_crop = 281
width_wide_crop = 201
# tyデータの高さ
height_ty = 201
# tyデータの幅
width_ty = 201
#-------------------------------------------------
#=================================================
# レイヤーの関数
#-------------------------------------------------
# 重みの計算
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
#-------------------------------------------------
#-------------------------------------------------
# バイアスの計算
def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))
#-------------------------------------------------
#-------------------------------------------------
# カーネルの初期化
def truncated_normal_var(name, shape, dtype):
    return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
#-------------------------------------------------
#-------------------------------------------------
# 2D Convolution + Relu layer
def conv2d_relu(input_image, kernel, bias, stride=None):
    conv = tf.nn.conv2d(input_image, kernel, strides=stride, padding='SAME')
    conv_bias = tf.nn.bias_add(conv, bias)
    conv_relu = tf.nn.relu(conv_bias)
    return conv_relu
#-------------------------------------------------
#-------------------------------------------------
# max-pooling layer
def max_pool(x, ksize=None, stride=None):
    return tf.nn.max_pool(x, ksize=ksize, strides=stride, padding='SAME')
#-------------------------------------------------
#-------------------------------------------------
# batch normalization
def batch_norm(inputs,training, trainable=False):
    res = tf.layers.batch_normalization(inputs, training=training, trainable=training)
    return res
#-------------------------------------------------
#=================================================
#=================================================
# CNN (ir_ty-stream)
def ir_ty_stream(input_image, training=False, reuse=False):
    conv_ir_ty1, conv_ir_ty2, conv_ir_ty3, conv_ir_ty4, conv_ir_ty5 = 0, 0, 0, 0, 0
    with tf.variable_scope('ir_ty_stream') as scope:
        if reuse:
            scope.reuse_variables()
        #input_image = tf.image.resize_image_with_crop_or_pad(input_image, 151,151)
        #-------------------------------------------------
        # １つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty1') as scope:
        # 畳み込みカーネル(size:5x5 channel:9 features:32 )
        conv_ir_ty_kernel1 = weight_variable(name='conv_ir_ty_kernel1', shape=[kernal_size, kernal_size, c_ty, c_cnn])
        # バイアス項を初期化
        conv_ir_ty_bias1 = bias_variable('conv_ir_ty_bias1', shape=[c_cnn])
        # 画像全体をストライド2で畳み込む
        conv_ir_ty1= conv2d_relu(input_image, conv_ir_ty_kernel1, conv_ir_ty_bias1, stride=[1,2,2,1])
        # batch normalization
        conv_ir_ty1 = batch_norm(conv_ir_ty1, training)
        #------------------------------------------------
        #-------------------------------------------------
        # 2つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty2') as scope:
        # 畳み込みカーネル(size:5x5 channel:32 features:64 )
        conv_ir_ty_kernel2 = weight_variable(name='conv_ir_ty_kernel2', shape=[kernal_size, kernal_size, c_cnn, c_cnn*2])
        # バイアス項を初期化
        conv_ir_ty_bias2 = bias_variable('conv_ir_ty_bias2', shape=[c_cnn*2])
        # 画像全体をストライド2で畳み込む
        conv_ir_ty2= conv2d_relu(conv_ir_ty1, conv_ir_ty_kernel2, conv_ir_ty_bias2, stride=[1,2,2,1])
        # batch normalization
        conv_ir_ty2 = batch_norm(conv_ir_ty2, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty3') as scope:
        # 畳み込みカーネル(size:5x5 channel:64 features:128 )
        conv_ir_ty_kernel3 = weight_variable(name='conv_ir_ty_kernel3', shape=[kernal_size, kernal_size, c_cnn*2, c_cnn*4])
        # バイアス項を初期化
        conv_ir_ty_bias3 = bias_variable('conv_ir_ty_bias3', shape=[c_cnn*4])
        # 画像全体をストライド2で畳み込む
        conv_ir_ty3= conv2d_relu(conv_ir_ty2, conv_ir_ty_kernel3, conv_ir_ty_bias3, stride=[1,2,2,1])
        # batch normalization
        conv_ir_ty3 = batch_norm(conv_ir_ty3, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 4つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty4') as scope:
        # 畳み込みカーネル(size:5x5 channel:128 features:256 )
        conv_ir_ty_kernel4 = weight_variable(name='conv_ir_ty_kernel4', shape=[kernal_size, kernal_size, c_cnn*4, c_cnn*8])
        # バイアス項を初期化
        conv_ir_ty_bias4 = bias_variable('conv_ir_ty_bias4', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_ir_ty4= conv2d_relu(conv_ir_ty3, conv_ir_ty_kernel4, conv_ir_ty_bias4, stride=[1,2,2,1])
        # batch normalization
        conv_ir_ty4 = batch_norm(conv_ir_ty4, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 5つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty5') as scope:
        # 畳み込みカーネル(size:5x5 channel:256 features:512 )
        conv_ir_ty_kernel5 = weight_variable(name='conv_ir_ty_kernel5', shape=[kernal_size, kernal_size, c_cnn*8, c_cnn*8])
        # バイアス項を初期化
        conv_ir_ty_bias5 = bias_variable('conv_ir_ty_bias5', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_ir_ty5= conv2d_relu(conv_ir_ty4, conv_ir_ty_kernel5, conv_ir_ty_bias5, stride=[1,2,2,1])
        # batch normalization
        conv_ir_ty5 = batch_norm(conv_ir_ty5, training)
        #-------------------------------------------------
        #conv_ir_ty5_size = np.prod(conv_ir_ty5.get_shape().as_list()[1:])
        s = conv_ir_ty5.get_shape().as_list()
        input_result = tf.reshape(conv_ir_ty5, [-1, s[1]*s[2]*s[3]])
        input_dim = input_result.get_shape().as_list()[1]
        print(type(input_dim))
        #-------------------------------------------------
        # 1つ目の全結合層
        #with tf.variable_scope('full1') as scope:
        # 重みを初期化
        full_weight1 = weight_variable(name='full_multi1', shape=[input_dim, input_dim//2])
        # バイアス項を初期化
        full_bias1 = bias_variable('full_bias1', shape=[input_dim//2])
        # 結合
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_result, full_weight1), full_bias1))
        #full_layer1 = tf.add(tf.matmul(input_result, full_weight1), full_bias1)
        #-------------------------------------------------
        #-------------------------------------------------
        # 2つ目の全結合層
        #with tf.variable_scope('full2') as scope:
        # 重みを初期化
        full_weight2 = weight_variable(name='full_multi2', shape=[input_dim//2, input_dim//4])
        # バイアス項を初期化
        full_bias2 = bias_variable('full_bias2', shape=[input_dim//4])
        # 結合
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
        #full_layer2 = tf.add(tf.matmul(full_layer1, full_weight2), full_bias2)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の全結合層
        #with tf.variable_scope('full3') as scope:
        # 重みを初期化
        full_weight3 = weight_variable(name='full_multi3', shape=[input_dim//4, 1])
        # バイアス項を初期化
        full_bias3 = bias_variable('full_bias3', shape=[1])
        # 結合
        full_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(full_layer2, full_weight3), full_bias3))
        #-------------------------------------------------
        return full_layer3, conv_ir_ty1, conv_ir_ty2, conv_ir_ty3, conv_ir_ty4, conv_ir_ty5, full_layer1, full_layer2
#=================================================
#=================================================
# CNN (ir_wide-stream)
def ir_wide_stream(input_image, training=False, reuse=False):
    conv_ir_wide1, conv_ir_wide2, conv_ir_wide3, conv_ir_wide4, conv_ir_wide5 = 0, 0, 0, 0, 0
    with tf.variable_scope('ir_wide_stream') as scope:
        if reuse:
            scope.reuse_variables()
        #input_image = tf.image.resize_image_with_crop_or_pad(input_image, 151,151)
        #-------------------------------------------------
        # １つ目の畳み込み層
        #with tf.variable_scope('conv_ir_wide1') as scope:
        # 畳み込みカーネル(size:5x5 channel:9 features:32 )
        conv_ir_wide_kernel1 = weight_variable(name='conv_ir_wide_kernel1', shape=[kernal_size, kernal_size, c_ty, c_cnn])
        # バイアス項を初期化
        conv_ir_wide_bias1 = bias_variable('conv_ir_wide_bias1', shape=[c_cnn])
        # 画像全体をストライド2で畳み込む
        conv_ir_wide1= conv2d_relu(input_image, conv_ir_wide_kernel1, conv_ir_wide_bias1, stride=[1,2,2,1])
        # batch normalization
        conv_ir_wide1 = batch_norm(conv_ir_wide1, training)
        #------------------------------------------------
        #-------------------------------------------------
        # 2つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty2') as scope:
        # 畳み込みカーネル(size:5x5 channel:32 features:64 )
        conv_ir_wide_kernel2 = weight_variable(name='conv_ir_wide_kernel2', shape=[kernal_size, kernal_size, c_cnn, c_cnn*2])
        # バイアス項を初期化
        conv_ir_wide_bias2 = bias_variable('conv_ir_wide_bias2', shape=[c_cnn*2])
        # 画像全体をストライド2で畳み込む
        conv_ir_wide2= conv2d_relu(conv_ir_wide1, conv_ir_wide_kernel2, conv_ir_wide_bias2, stride=[1,2,2,1])
        # batch normalization
        conv_ir_wide2 = batch_norm(conv_ir_wide2, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty3') as scope:
        # 畳み込みカーネル(size:5x5 channel:64 features:128 )
        conv_ir_wide_kernel3 = weight_variable(name='conv_ir_wide_kernel3', shape=[kernal_size, kernal_size, c_cnn*2, c_cnn*4])
        # バイアス項を初期化
        conv_ir_wide_bias3 = bias_variable('conv_ir_wide_bias3', shape=[c_cnn*4])
        # 画像全体をストライド2で畳み込む
        conv_ir_wide3= conv2d_relu(conv_ir_wide2, conv_ir_wide_kernel3, conv_ir_wide_bias3, stride=[1,2,2,1])
        # batch normalization
        conv_ir_wide3 = batch_norm(conv_ir_wide3, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 4つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty4') as scope:
        # 畳み込みカーネル(size:5x5 channel:128 features:256 )
        conv_ir_wide_kernel4 = weight_variable(name='conv_ir_wide_kernel4', shape=[kernal_size, kernal_size, c_cnn*4, c_cnn*8])
        # バイアス項を初期化
        conv_ir_wide_bias4 = bias_variable('conv_ir_wide_bias4', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_ir_wide4= conv2d_relu(conv_ir_wide3, conv_ir_wide_kernel4, conv_ir_wide_bias4, stride=[1,2,2,1])
        # batch normalization
        conv_ir_wide4 = batch_norm(conv_ir_wide4, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 5つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty5') as scope:
        # 畳み込みカーネル(size:5x5 channel:256 features:512 )
        conv_ir_wide_kernel5 = weight_variable(name='conv_ir_wide_kernel5', shape=[kernal_size, kernal_size, c_cnn*8, c_cnn*8])
        # バイアス項を初期化
        conv_ir_wide_bias5 = bias_variable('conv_ir_wide_bias5', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_ir_wide5= conv2d_relu(conv_ir_wide4, conv_ir_wide_kernel5, conv_ir_wide_bias5, stride=[1,2,2,1])
        # batch normalization
        conv_ir_wide5 = batch_norm(conv_ir_wide5, training)
        #-------------------------------------------------
        #conv_ir_ty5_size = np.prod(conv_ir_ty5.get_shape().as_list()[1:])
        s = conv_ir_wide5.get_shape().as_list()
        input_result = tf.reshape(conv_ir_wide5, [-1, s[1]*s[2]*s[3]])
        input_dim = input_result.get_shape().as_list()[1]
        print(type(input_dim))
        #-------------------------------------------------
        # 1つ目の全結合層
        #with tf.variable_scope('full1') as scope:
        # 重みを初期化
        full_weight1 = weight_variable(name='full_multi1', shape=[input_dim, input_dim//2])
        # バイアス項を初期化
        full_bias1 = bias_variable('full_bias1', shape=[input_dim//2])
        # 結合
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_result, full_weight1), full_bias1))
        #full_layer1 = tf.add(tf.matmul(input_result, full_weight1), full_bias1)
        #-------------------------------------------------
        #-------------------------------------------------
        # 2つ目の全結合層
        #with tf.variable_scope('full2') as scope:
        # 重みを初期化
        full_weight2 = weight_variable(name='full_multi2', shape=[input_dim//2, input_dim//4])
        # バイアス項を初期化
        full_bias2 = bias_variable('full_bias2', shape=[input_dim//4])
        # 結合
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
        #full_layer2 = tf.add(tf.matmul(full_layer1, full_weight2), full_bias2)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の全結合層
        #with tf.variable_scope('full3') as scope:
        # 重みを初期化
        full_weight3 = weight_variable(name='full_multi3', shape=[input_dim//4, 1])
        # バイアス項を初期化
        full_bias3 = bias_variable('full_bias3', shape=[1])
        # 結合
        full_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(full_layer2, full_weight3), full_bias3))
        #-------------------------------------------------
        return full_layer3, conv_ir_wide1, conv_ir_wide2, conv_ir_wide3, conv_ir_wide4, conv_ir_wide5, full_layer1, full_layer2
#=================================================
#=================================================
# CNN (wv_ty-stream)
def wv_ty_stream(input_image, training=False, reuse=False):
    conv_wv_ty1, conv_wv_ty2, conv_wv_ty3, conv_wv_ty4, conv_wv_ty5 = 0, 0, 0, 0, 0
    with tf.variable_scope('wv_ty_stream') as scope:
        if reuse:
            scope.reuse_variables()
        #input_image = tf.image.resize_image_with_crop_or_pad(input_image, 151,151)
        #-------------------------------------------------
        # １つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty1') as scope:
        # 畳み込みカーネル(size:5x5 channel:9 features:32 )
        conv_wv_ty_kernel1 = weight_variable(name='conv_wv_ty_kernel1', shape=[kernal_size, kernal_size, c_ty, c_cnn])
        # バイアス項を初期化
        conv_wv_ty_bias1 = bias_variable('conv_wv_ty_bias1', shape=[c_cnn])
        # 画像全体をストライド2で畳み込む
        conv_wv_ty1= conv2d_relu(input_image, conv_wv_ty_kernel1, conv_wv_ty_bias1, stride=[1,2,2,1])
        # batch normalization
        conv_wv_ty1 = batch_norm(conv_wv_ty1, training)
        #------------------------------------------------
        #-------------------------------------------------
        # 2つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty2') as scope:
        # 畳み込みカーネル(size:5x5 channel:32 features:64 )
        conv_wv_ty_kernel2 = weight_variable(name='conv_wv_ty_kernel2', shape=[kernal_size, kernal_size, c_cnn, c_cnn*2])
        # バイアス項を初期化
        conv_wv_ty_bias2 = bias_variable('conv_wv_ty_bias2', shape=[c_cnn*2])
        # 画像全体をストライド2で畳み込む
        conv_wv_ty2= conv2d_relu(conv_wv_ty1, conv_wv_ty_kernel2, conv_wv_ty_bias2, stride=[1,2,2,1])
        # batch normalization
        conv_wv_ty2 = batch_norm(conv_wv_ty2, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty3') as scope:
        # 畳み込みカーネル(size:5x5 channel:64 features:128 )
        conv_wv_ty_kernel3 = weight_variable(name='conv_wv_ty_kernel3', shape=[kernal_size, kernal_size, c_cnn*2, c_cnn*4])
        # バイアス項を初期化
        conv_wv_ty_bias3 = bias_variable('conv_wv_ty_bias3', shape=[c_cnn*4])
        # 画像全体をストライド2で畳み込む
        conv_wv_ty3= conv2d_relu(conv_wv_ty2, conv_wv_ty_kernel3, conv_wv_ty_bias3, stride=[1,2,2,1])
        # batch normalization
        conv_wv_ty3 = batch_norm(conv_wv_ty3, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 4つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty4') as scope:
        # 畳み込みカーネル(size:5x5 channel:128 features:256 )
        conv_wv_ty_kernel4 = weight_variable(name='conv_wv_ty_kernel4', shape=[kernal_size, kernal_size, c_cnn*4, c_cnn*8])
        # バイアス項を初期化
        conv_wv_ty_bias4 = bias_variable('conv_wv_ty_bias4', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_wv_ty4= conv2d_relu(conv_wv_ty3, conv_wv_ty_kernel4, conv_wv_ty_bias4, stride=[1,2,2,1])
        # batch normalization
        conv_wv_ty4 = batch_norm(conv_wv_ty4, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 5つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty5') as scope:
        # 畳み込みカーネル(size:5x5 channel:256 features:512 )
        conv_wv_ty_kernel5 = weight_variable(name='conv_wv_ty_kernel5', shape=[kernal_size, kernal_size, c_cnn*8, c_cnn*8])
        # バイアス項を初期化
        conv_wv_ty_bias5 = bias_variable('conv_wv_ty_bias5', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_wv_ty5= conv2d_relu(conv_wv_ty4, conv_wv_ty_kernel5, conv_wv_ty_bias5, stride=[1,2,2,1])
        # batch normalization
        conv_wv_ty5 = batch_norm(conv_wv_ty5, training)
        #-------------------------------------------------
        #conv_ir_ty5_size = np.prod(conv_ir_ty5.get_shape().as_list()[1:])
        s = conv_wv_ty5.get_shape().as_list()
        input_result = tf.reshape(conv_wv_ty5, [-1, s[1]*s[2]*s[3]])
        input_dim = input_result.get_shape().as_list()[1]
        print(type(input_dim))
        #-------------------------------------------------
        # 1つ目の全結合層
        #with tf.variable_scope('full1') as scope:
        # 重みを初期化
        full_weight1 = weight_variable(name='full_multi1', shape=[input_dim, input_dim//2])
        # バイアス項を初期化
        full_bias1 = bias_variable('full_bias1', shape=[input_dim//2])
        # 結合
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_result, full_weight1), full_bias1))
        #full_layer1 = tf.add(tf.matmul(input_result, full_weight1), full_bias1)
        #-------------------------------------------------
        #-------------------------------------------------
        # 2つ目の全結合層
        #with tf.variable_scope('full2') as scope:
        # 重みを初期化
        full_weight2 = weight_variable(name='full_multi2', shape=[input_dim//2, input_dim//4])
        # バイアス項を初期化
        full_bias2 = bias_variable('full_bias2', shape=[input_dim//4])
        # 結合
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
        #full_layer2 = tf.add(tf.matmul(full_layer1, full_weight2), full_bias2)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の全結合層
        #with tf.variable_scope('full3') as scope:
        # 重みを初期化
        full_weight3 = weight_variable(name='full_multi3', shape=[input_dim//4, 1])
        # バイアス項を初期化
        full_bias3 = bias_variable('full_bias3', shape=[1])
        # 結合
        full_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(full_layer2, full_weight3), full_bias3))
        #-------------------------------------------------
        return full_layer3, conv_wv_ty1, conv_wv_ty2, conv_wv_ty3, conv_wv_ty4, conv_wv_ty5, full_layer1, full_layer2
#=================================================
#=================================================
# CNN (diff_ty-stream)
def diff_ty_stream(input_image, training=False, reuse=False):
    conv_diff_ty1, conv_diff_ty2, conv_diff_ty3, conv_diff_ty4, conv_diff_ty5 = 0, 0, 0, 0, 0
    with tf.variable_scope('ir_ty_stream') as scope:
        if reuse:
            scope.reuse_variables()
        #input_image = tf.image.resize_image_with_crop_or_pad(input_image, 151,151)
        #-------------------------------------------------
        # １つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty1') as scope:
        # 畳み込みカーネル(size:5x5 channel:9 features:32 )
        conv_diff_ty_kernel1 = weight_variable(name='conv_diff_ty_kernel1', shape=[kernal_size, kernal_size, c_ty, c_cnn])
        # バイアス項を初期化
        conv_diff_ty_bias1 = bias_variable('conv_diff_ty_bias1', shape=[c_cnn])
        # 画像全体をストライド2で畳み込む
        conv_diff_ty1= conv2d_relu(input_image, conv_diff_ty_kernel1, conv_diff_ty_bias1, stride=[1,2,2,1])
        # batch normalization
        conv_diff_ty1 = batch_norm(conv_diff_ty1, training)
        #------------------------------------------------
        #-------------------------------------------------
        # 2つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty2') as scope:
        # 畳み込みカーネル(size:5x5 channel:32 features:64 )
        conv_diff_ty_kernel2 = weight_variable(name='conv_diff_ty_kernel2', shape=[kernal_size, kernal_size, c_cnn, c_cnn*2])
        # バイアス項を初期化
        conv_diff_ty_bias2 = bias_variable('conv_diff_ty_bias2', shape=[c_cnn*2])
        # 画像全体をストライド2で畳み込む
        conv_diff_ty2= conv2d_relu(conv_diff_ty1, conv_diff_ty_kernel2, conv_diff_ty_bias2, stride=[1,2,2,1])
        # batch normalization
        conv_diff_ty2 = batch_norm(conv_diff_ty2, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty3') as scope:
        # 畳み込みカーネル(size:5x5 channel:64 features:128 )
        conv_diff_ty_kernel3 = weight_variable(name='conv_diff_ty_kernel3', shape=[kernal_size, kernal_size, c_cnn*2, c_cnn*4])
        # バイアス項を初期化
        conv_diff_ty_bias3 = bias_variable('conv_diff_ty_bias3', shape=[c_cnn*4])
        # 画像全体をストライド2で畳み込む
        conv_diff_ty3= conv2d_relu(conv_diff_ty2, conv_diff_ty_kernel3, conv_diff_ty_bias3, stride=[1,2,2,1])
        # batch normalization
        conv_diff_ty3 = batch_norm(conv_diff_ty3, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 4つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty4') as scope:
        # 畳み込みカーネル(size:5x5 channel:128 features:256 )
        conv_diff_ty_kernel4 = weight_variable(name='conv_diff_ty_kernel4', shape=[kernal_size, kernal_size, c_cnn*4, c_cnn*8])
        # バイアス項を初期化
        conv_diff_ty_bias4 = bias_variable('conv_diff_ty_bias4', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_diff_ty4= conv2d_relu(conv_diff_ty3, conv_diff_ty_kernel4, conv_diff_ty_bias4, stride=[1,2,2,1])
        # batch normalization
        conv_diff_ty4 = batch_norm(conv_diff_ty4, training)
        #-------------------------------------------------
        #-------------------------------------------------
        # 5つ目の畳み込み層
        #with tf.variable_scope('conv_ir_ty5') as scope:
        # 畳み込みカーネル(size:5x5 channel:256 features:512 )
        conv_diff_ty_kernel5 = weight_variable(name='conv_diff_ty_kernel5', shape=[kernal_size, kernal_size, c_cnn*8, c_cnn*8])
        # バイアス項を初期化
        conv_diff_ty_bias5 = bias_variable('conv_diff_ty_bias5', shape=[c_cnn*8])
        # 画像全体をストライド2で畳み込む
        conv_diff_ty5= conv2d_relu(conv_diff_ty4, conv_diff_ty_kernel5, conv_diff_ty_bias5, stride=[1,2,2,1])
        # batch normalization
        conv_diff_ty5 = batch_norm(conv_diff_ty5, training)
        #-------------------------------------------------
        #conv_ir_ty5_size = np.prod(conv_ir_ty5.get_shape().as_list()[1:])
        s = conv_diff_ty5.get_shape().as_list()
        input_result = tf.reshape(conv_diff_ty5, [-1, s[1]*s[2]*s[3]])
        input_dim = input_result.get_shape().as_list()[1]
        print(type(input_dim))
        #-------------------------------------------------
        # 1つ目の全結合層
        #with tf.variable_scope('full1') as scope:
        # 重みを初期化
        full_weight1 = weight_variable(name='full_multi1', shape=[input_dim, input_dim//2])
        # バイアス項を初期化
        full_bias1 = bias_variable('full_bias1', shape=[input_dim//2])
        # 結合
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_result, full_weight1), full_bias1))
        #full_layer1 = tf.add(tf.matmul(input_result, full_weight1), full_bias1)
        #-------------------------------------------------
        #-------------------------------------------------
        # 2つ目の全結合層
        #with tf.variable_scope('full2') as scope:
        # 重みを初期化
        full_weight2 = weight_variable(name='full_multi2', shape=[input_dim//2, input_dim//4])
        # バイアス項を初期化
        full_bias2 = bias_variable('full_bias2', shape=[input_dim//4])
        # 結合
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
        #full_layer2 = tf.add(tf.matmul(full_layer1, full_weight2), full_bias2)
        #-------------------------------------------------
        #-------------------------------------------------
        # 3つ目の全結合層
        #with tf.variable_scope('full3') as scope:
        # 重みを初期化
        full_weight3 = weight_variable(name='full_multi3', shape=[input_dim//4, 1])
        # バイアス項を初期化
        full_bias3 = bias_variable('full_bias3', shape=[1])
        # 結合
        full_layer3 = tf.nn.sigmoid(tf.add(tf.matmul(full_layer2, full_weight3), full_bias3))
        #-------------------------------------------------
        return full_layer3, conv_diff_ty1, conv_diff_ty2, conv_diff_ty3, conv_diff_ty4, conv_diff_ty5, full_layer1, full_layer2
#=================================================
if __name__=='__main__':
    # tensorflowの設定
    # allow_growth = True->必要になったらメモリ確保, False->全部使う
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    #====================================================================
    # メイン
    # batch normalizationを実行した後のパラメータを更新
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="ir_ty_stream")
    with tf.control_dependencies(extra_update_ops):
        Gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ir_ty_stream")
        # 最適化関数
        my_opt = tf.train.AdamOptimizer(l_rate)
        train_step_cross_entropy = my_opt.minimize(loss_cross_entropy_train)
    # 変数の初期化
    init = tf.global_variables_initializer()
    sess.run(init)
    #saver = tf.train.Saver()
