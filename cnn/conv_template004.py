'''
駄文です。
'''

import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
from sklearn.datasets import fetch_mldata

class cnn:
    def __init__(self, x, y):
        #placeholder
        self.x = tf.reshape(x, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, shape=(None, 10))
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
        conv1_2d = tf.nn.conv2d(self.x, conv1_Filter, strides=[1, 1, 1, 1], padding='SAME')
        # バイアスの加算
        conv1_b_a = tf.nn.bias_add(conv1_2d, bias1)
        # RELU関数
        conv1_fin = tf.nn.relu(conv1_b_a)

        '''
        #------conv2------#
        '''
        # 畳み込み層の次元数
        conv2_dim = 20
        conv2_Filter = tf.Variable(tf.random_normal([5, 5, conv1_dim, conv2_dim], stddev=0.1), dtype=tf.float32)
        bias2 = tf.Variable(tf.random_normal([conv2_dim]))
        conv2_2d = tf.nn.conv2d(conv1_fin, conv2_Filter, strides=[1, 1, 1, 1], padding='SAME')
        conv2_b_a = tf.nn.bias_add(conv2_2d, bias2)
        conv2_fin = tf.nn.relu(conv2_b_a)

        #pdb.set_trace()

        '''
        #------conv3------#
        '''
        # 畳み込み層の次元数
        conv3_dim = 20
        conv3_Filter = tf.Variable(tf.random_normal([5, 5, conv2_dim, conv3_dim], stddev=0.1), dtype=tf.float32)
        bias3 = tf.Variable(tf.random_normal([conv3_dim]))
        conv3_2d = tf.nn.conv2d(conv2_fin, conv3_Filter, strides=[1, 1, 1, 1], padding='SAME')
        conv3_b_a = tf.nn.bias_add(conv3_2d, bias3)
        conv3_fin = tf.nn.relu(conv3_b_a)

        #pdb.set_trace()

        '''
        #------conv4------#
        '''
        # 畳み込み層の次元数
        conv4_dim = 20
        conv4_Filter = tf.Variable(tf.random_normal([5, 5, conv3_dim, conv4_dim], stddev=0.1), dtype=tf.float32)
        bias4 = tf.Variable(tf.random_normal([conv4_dim]))
        conv4_2d = tf.nn.conv2d(conv3_fin, conv4_Filter, strides=[1, 1, 1, 1], padding='SAME')
        conv4_b_a = tf.nn.bias_add(conv4_2d, bias4)
        conv4_fin = tf.nn.relu(conv4_b_a)

        '''
        #------conv5------#
        '''
        # 畳み込み層の次元数
        conv5_dim = 20
        conv5_Filter = tf.Variable(tf.random_normal([5, 5, conv4_dim, conv5_dim], stddev=0.1), dtype=tf.float32)
        bias5 = tf.Variable(tf.random_normal([conv5_dim]))
        conv5_2d = tf.nn.conv2d(conv4_fin, conv5_Filter, strides=[1, 1, 1, 1], padding='SAME')
        conv5_b_a = tf.nn.bias_add(conv5_2d, bias5)
        conv5_fin = tf.nn.relu(conv5_b_a)

        #pdb.set_trace()


        '''
        # 大きなファイルになる場合は畳み込み層をさらに増やした方が良い
        # ) 5層くらい
        '''

        '''
        #------fully connected 1------#
        # 畳み込み層と同じような形式で記述するようにする
        '''
        # 全結合層1
        # 1次元データに圧縮
        fully1_flat = tf.reshape(conv5_fin,[-1])
        # 全結合層1の次元数
        fully1_dim = 512
        #全結合層1の処理
        input_size = x_.shape[1] * x_.shape[2]
        weight_f1 = tf.Variable(tf.random_normal([input_size,fully1_dim], stddev=0.1), dtype=tf.float32)
        bias_f1 = tf.Variable(tf.random_normal([fully1_dim]))
        fc1 = tf.nn.relu(tf.matmul(fully1_flat, weight_f1)+bias_f1)
        #fully1 = tf.contrib.layers.fully_connected(fully1_flat, fully1_dim)
        pdb.set_trace()
        '''
        #------fully connected 2------#
        # 畳み込み層と同じような形式で記述するようにする
        '''
        # 全結合層2
        # 全結合層2の次元数
        fully2_dim = 256
        #全結合層2の処理
        weight_f2 = tf.Variable(tf.random_normal([fully1_dim,fully1_dim], stddev=0.1), dtype=tf.float32)
        bias_f2 = tf.Variable(tf.random_normal([fully2_dim]))
        fc2 = tf.nn.relu(tf.matmul(fc1, weight_f2)+bias_f2)

        '''
        #------fully connected 3------#
        '''
        # 全結合層3
        # 全結合層3の次元数
        fully3_dim = 10
        # 全結合層2の処理
        weight_f3 = tf.Variable(tf.random_normal([fully1_dim,10], stddev=0.1), dtype=tf.float32)
        bias_f3 = tf.Variable(tf.random_normal([fully2_dim]))
        fc3 = tf.nn.relu(tf.matmul(fc2, weight_f3)+bias_f3)
        # 代入してるだけ
        conv_last = fc3

    '''
    #------loss------#
    '''
    def loss(self,y_):
        
        # 交差エントロピー
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, y_)
        # 損失関数
        return tf.reduce_mean(cross_entropy)

    '''
    #------accuracy------#
    '''
    def accuracy(self,y_):
    # 正解率の計算
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.y, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    # 全結合層：3つ（うち出力層1つ）
    # 畳み込み層：5つ
    '''

if __name__ == "__main__":

    # 学習
    EPOCH_NUM = 5
    BATCH_SIZE = 1000

    # 教師データ
    mnist = fetch_mldata('MNIST original', data_home='.')
    mnist.data = mnist.data.astype(np.float32) # 画像データ　784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...], ... ]
    mnist.data /= 255 # 0-1に正規化する
    mnist.target = mnist.target.astype(np.int32) # ラベルデータ70000
 
    # 教師データを変換
    N = 60000
    train_x, test_x = np.split(mnist.data,   [N]) # 教師データ
    train_y, test_y = np.split(mnist.target, [N]) # テスト用のデータ
    train_x = train_x.reshape((len(train_x), 28, 28, 1)) # (N, height, width, channel)
    test_x = test_x.reshape((len(test_x), 28, 28, 1))
    # ラベルはone-hotベクトルに変換する
    train_y = np.eye(np.max(train_y)+1)[train_y]
    test_y = np.eye(np.max(test_y)+1)[test_y]

    # 学習
    print("Train")
    with tf.Session() as sess:
        st = time.time()
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCH_NUM):
            perm = np.random.permutation(N)
            total_loss = 0
            for i in range(0, N, BATCH_SIZE):
                batch_x = train_x[perm[i:i+BATCH_SIZE]]
                batch_y = train_y[perm[i:i+BATCH_SIZE]]
                total_loss += cross_entropy.eval(feed_dict={x_: batch_x, y_: batch_y})
                train_step.run(feed_dict={x_: batch_x, y_: batch_y})
            test_accuracy = accuracy.eval(feed_dict={x_: test_x, y_: test_y})
            if (epoch+1) % 1 == 0:
                ed = time.time()
                print("epoch:\t{}\ttotal loss:\t{}\tvaridation accuracy:\t{}\ttime:\t{}".format(epoch+1, total_loss, test_accuracy, ed-st))
                st = time.time()
    



        























