# -*- Pythonによる機械学習1 宿題2(2) -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------
# クラスの定義始まり
class sentence:
    dataPath = "sentiment_labelled_sentences" #データのフォルダ名
    
    #--------------------
    # CSVファイルの読み込み
    # fname: ファイルパス（文字列）
    def __init__(self,fname):
        
        # ファイルのパス設定
        fullpath = os.path.join(self.dataPath,fname)
        
        # csv形式のデータ読み込み
        self.data = pd.read_csv(fullpath,"\t")
    #-----------------------------
    
    #-----------------------------
    # 文字列検索
    # keyword: 検索キーワード（文字列）
    def search(self,keyword):
        # sentence列で、keywordを含む要素のインデックスを取得
        results = self.data["sentence"].str.contains(keyword)
        
        # np.arrayとして返す
        return self.data["sentence"][results].values
    #-----------------------------
    
    #-----------------------------
    # スコアが「1」の商品レビューの文章を、numpy.arrayの各要素に格納して返す
    def getPositiveSentence(self):
        results = self.data["score"] == 1
        return self.data["sentence"][results].values
    #-----------------------------
    
    #-----------------------------
    # keywordを含むインデックスのスコアを返す
    def plotScoreRatio(self,keyword):
        index = self.data["sentence"].str.contains(keyword)
        return self.data["score"][index].values

# クラスの定義終わり
#--------------------

#--------------------
# メインの始まり
if __name__ == "__main__":
    # データファイルamazon_cells_labelled.txtを指定して、インスタンス化
    myData = sentence("amazon_cells_labelled.txt")
    
    # 検索
    # results = myData.search("very good")
    
    # 検索結果内のスコアが「1」の商品レビュー数
    results = myData.plotScoreRatio("not")
    ratio = np.array([0,1])
    
    # スコア「0」の割合：negative スコア「1」の割合：positive
    negative = len(results[[results[ind] == 0 for ind in np.arange(len(results))]]) / len(results)
    positive = len(results[[results[ind] == 1 for ind in np.arange(len(results))]]) / len(results)
    
    # グラフプロット
    plt.bar(ratio,[negative,positive],tick_label=["negative","positive"],align="center")
    
    # タイトル：title("タイトル名", "データ名2", fontsize=フォントサイズ)
    plt.title("keyword:not",fontsize=14)

    # 各軸のラベル：xlabel("ラベル名", fontsize=フォントサイズ)
    plt.xlabel("Score",fontsize=14)
    plt.ylabel("Ratio",fontsize=14)
    
    # グラフを表示
    plt.show()
# メインの終わり
#--------------------
