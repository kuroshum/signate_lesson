# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------
# クラスの定義始まり
class sentence:
	dataPath = 'sentiment_labelled_sentences'
	def __init__(self,fname):
		fullpath = os.path.join(self.dataPath,fname)
		self.data = pd.read_csv(fullpath,'\t')
	
	def search(self, keyword):
		results = self.data['sentence'].str.contains(keyword)
		return self.data['sentence'][results].values
	
	def getPositiveSentence(self):
		s = self.data['score'].values == 1
		return self.data['sentence'][s].values

	def plotScoreRatio(self,key):
		target = self.data['sentence'].str.contains(key)
		target2 = self.data['score'][target].values
		print(len(target))
		print(len(target2))
		h = self.data['score'][target2].values
		num1 = np.array([])
		num0 = np.array([])
		for i in np.arange(len(h)):
			if h[i] == 1:
				num1 = np.append(num1,h[i])
			else:
				num0 = np.append(num0,h[i])
		print(len(num1))
		print(len(num0))
		left = np.array(['Positive','Negative'])
		num1 = len(num1) / len(target2)
		num0 = len(num0) / len(target2)
		height = np.array([num1,num0])
		label = ["positive","negative"]
		plt.bar(left,height,tick_label = left, align = "center")
		plt.title("keyword:"+key)
		plt.xlabel('Score')
		plt.ylabel('Ratio')

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
	d = myData.getPositiveSentence()
	myData.plotScoreRatio("not")

	# 検索結果の表示
	for ind in np.arange(len(d)):
		print(ind,":",d[ind])
#メインの終わり
#-------------------