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

	#------------------------------------
	# 
	def getPositiveSentence(self):
		result = self.data[self.data['score']==1]['sentence']
		return result
	#------------------------------------

	#------------------------------------
	def plotScoreRatio(self,keyword):
		s = self.data[self.data['sentence'].str.contains(keyword)]
		n0 = len(s[s['score']==0]['sentence'])
		n1 = len(s[s['score']==1]['sentence'])
		height = np.array([n0,n1])/len(s)

		left = np.array(['negative','positive'])
		plt.bar(left,height,align='center')
		plt.title("keyword:"+keyword,fontsize=14)
		plt.xlabel("Score",fontsize=14)
		plt.ylabel("Ratio",fontsize=14)
		plt.show()
#------------------------------------

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == '__main__':
	# データファイルamazon_cells_labelled.txtを指定して、インスタンス化
	myData = sentence('amazon_cells_labelled.txt')
	# 検索
	results = myData.search('very good')

	a = myData.getPositiveSentence()
	print(a)
	
	myData.plotScoreRatio('not')

	# 検索結果の表示
	for ind in np.arange(len(results)):
		print(ind,':',results[ind])
#メインの終わり
#-------------------