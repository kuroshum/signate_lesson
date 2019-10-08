# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#-------------------
# クラスの定義始まり
class sentence:
	dataPath = 'sentiment_labelled_sentences'    # データのフォルダ名
	
	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self, fname):
	
		# ファイルのパス設定
		fullpath = os.path.join(self.dataPath, fname)

		# csv形式のデータ読み込み
		self.data = pd.read_csv(fullpath, '\t')
	#------------------------------------

	#------------------------------------
	# 文字列検索
	# keyword: 検索キーワード（文字列）
	def search(self, keyword):
		# sentence列で、keywordを含む要素のインデックスを取得
		results = self.data['sentence'].str.contains(keyword)
		
		# np.arrayとして返す
		return self.data['sentence'][results].values
	#------------------------------------

	def getPositiveSentence(self):
		return self.data['sentence'][[self.data['score'][ind] == 1 for ind in np.arange(len(self.data['score']))]].values

	def plotScoreRatio(self, keyword):
		results1 = self.search(keyword)
		results2 = self.getPositiveSentence()
		value = len(set(results1) & set(results2)) / len(results1)
		labels = np.array(["negative", "positive"])
		plt.bar(labels, np.array([1 - value, value]), tick_label = labels, align = "center")
		plt.title(f"keyword:{keyword}")
		plt.xlabel("Score")
		plt.ylabel("Ratio")
		plt.savefig("sentenceData.png")
		plt.show()

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence("amazon_cells_labelled.txt")

	# 検索
	results = myData.search("very good")
	
	# 検索結果の表示
	for ind in np.arange(len(results)):
		print(ind, ":", results[ind])

	results = myData.getPositiveSentence()
	for ind in np.arange(len(results)):
		print(ind, ":", results[ind])

	myData.plotScoreRatio("not")

#メインの終わり
#-------------------
