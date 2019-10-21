# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#-------------------
# クラスの定義始まり
class sentence:
	dataPath = 'sentiment_labelled_sentences'    # データのフォルダ名
	
	#------------------------------------
	# CSVファイルの読み込み
	# fname: ファイルパス（文字列）
	def __init__(self,fname):
	
		# ファイルのパス設定
		fullpath = os.path.join(self.dataPath,fname)

		# csv形式のデータ読み込み
		self.data = pd.read_csv(fullpath,'\t')
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
		
		results = self.data[self.data['score']==1]['sentence']
		return results

			
	def plotScoreRatio(self,keyword):
		key = self.data['sentence'].str.contains(keyword)
		all = np.sum(key)
		ps = np.sum(key[self.data['score']==1])
		ng = np.sum(key[self.data['score']==0])

		ratio_p = ps/all
		ratio_n = ng/all

		ratio = np.array([ratio_n,ratio_p])
		label = ["negative","positive"]
		left = np.array([1,2])

		plt.bar(left,ratio,tick_label = label,align = "center")
		plt.xlabel("Score")
		plt.ylabel("Ratio")
		plt.title("keyword:"+ keyword)
		plt.show()
		
		print(all)
		print(ps)
		print(ng)


# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence("amazon_cells_labelled.txt")

	# 検索
	results = myData.search("very good")
	results2 = myData.getPositiveSentence()
	myData.plotScoreRatio("not")
	# 検索結果の表示
	for ind in np.arange(len(results)):
		print(ind,":",results[ind])
#メインの終わり
#-------------------