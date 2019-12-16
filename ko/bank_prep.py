import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score

bank_df = pd.read_csv('data/bank_prep.csv', sep=',')
# print(bank_df.head())

X = np.array(bank_df.drop('y', axis=1))
Y = np.array(bank_df[['y']])
print(np.sum(Y==1), np.sum(Y==0))

sampler = RandomUnderSampler(random_state=42)
X,Y = sampler.fit_resample(X,Y)
print(np.sum(Y==1), np.sum(Y==0))

# KFoldを使って交差検証
# 一つ目の引数はデータセットを分割する個数
# 二つ目の引数はデータセットをシャッフルするよう指定
kf = KFold(n_splits=18, shuffle=True)
scores = []

# 訓練データとテストデータの組み合わせを変えながら、モデルを作成し、精度を確認
for train_id, test_id in kf.split(X):
    # 訓練データを抽出
    x = X[train_id]
    y = Y[train_id]
    
    # 分類のための決定木インスタンスcifを作成
    cif = tree.DecisionTreeClassifier()
    # 訓練データを使って決定木モデルを作成
    # モデル作成にはデフォルトのパラメータをそのまま使用
    cif.fit(x,y)
    # predictを使って作成したモデルにテストデータを適用し出力を得る
    pred_y = cif.predict(X[test_id])
    # accuracy_scoreを使って、出力と正解の正誤数からモデルの精度を計算
    score = accuracy_score(Y[test_id], pred_y)
    scores.append(scores)

scores = np.array(scores)
print(scores.mean(), scores.std())
