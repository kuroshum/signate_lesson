# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

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
		#ap = self.data['score'].values
		s = self.data['sentence'].str.contains(self.data['score'].values==1)

		return self.data['sentence'][s].values

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence("amazon_cells_labelled.txt")

	# 検索
	results = myData.search("very good")
	s = myData.getPositiveSentence()

	# 検索結果の表示
	for ind in np.arange(len(s)):
		print(ind,":",s[ind])
#メインの終わり
#-------------------