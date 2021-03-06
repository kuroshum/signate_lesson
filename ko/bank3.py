import pandas as pd
import numpy as np
import pdb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

bank_df = pd.read_csv('data/bank_prep.csv', sep=',')
print(bank_df.head())

X = np.array(bank_df.drop('y', axis=1))
Y = np.array(bank_df[['y']])
print(np.sum(Y==1), np.sum(Y==0))

sampler = RandomUnderSampler(random_state=42)
X,Y = sampler.fit_resample(X,Y)
print(np.sum(Y==1), np.sum(Y==0))

# 特徴量を5つ選択
selector = SelectKBest(k=5)
selector.fit(X, Y)
mask = selector.get_support()
# どの変数を選択したかを確認
print(bank_df.drop('y', axis=1).columns)
print(mask)

# kFoldを使って交差検証
# 一つ目の引数はデータセットを分割する個数
# 二つ目の引数はデータセットをシャッフルするよう指定
kf = KFold(n_splits=18, shuffle=True)
scores = []

# 訓練データとテストデータの組み合わせを変えながら、モデルを作成し、精度を確認していきます。
for train_id, test_id in kf.split(X):
  # 訓練データを抽出
  x = X[train_id]
  y = Y[train_id]
  # 分類のための決定木インスタンスclfを生成します。
  cif = tree.DecisionTreeClassifier()
  # 訓練データを使って決定木モデルを作成
  # モデル作成には、デフォルトのパラメータをそのまま使用します。
  cif.fit(x,y)
  # predictを使って作成したモデルにテストデータを適用し出力を得ます
  pred_y = cif.predict(X[test_id])
  # accuracy_scoreを使って、出力と正解の正誤数からモデルの精度を計算します
  score = accuracy_score(Y[test_id], pred_y)
  scores.append(score)

scores = np.array(scores)
print(scores.mean(), scores.std())

print(recall_score(Y[test_id], pred_y))
print(precision_score(Y[test_id], pred_y))

print(cif)

# 探索するパラメータとその範囲を設定します。ここでは、分割の指標criterionはentropy(情報ゲイン)に固定し、
# 木の最大深さmax_depthは2~10、葉の最小データ数min_samples_leafは10~50にの間とします。
params = {
'criterion' : ['entropy'],
'max_depth' : [2,4,6,8,10],
'min_samples_leaf' : [10,20,30,40,50]
}

# グリッドサーチを実行する条件をセットします。
# 1つ目の引数にはアルゴリズムとして分類の決定木を、2つ目の引数に探索するパラメータセットを、
# 3つ目の引数には検証方法としてk-Fold交差検証を、4つ目の引数には評価方法としてモデル精度を指定します。
clf_gs = GridSearchCV(tree.DecisionTreeClassifier(), params, cv=KFold(n_splits=10,shuffle=True), scoring='accuracy')

clf_gs.fit(X,Y)

print(clf_gs.best_score_)
print(clf_gs.best_params_)

clf_best = tree.DecisionTreeClassifier(
criterion='entropy', max_depth=10, min_samples_leaf=20
)
clf_best.fit(X,Y)

print(clf_best.feature_importances_)

# 影響度の高さを降順でソートし、そのインデックスを得る
ind = np.argsort(clf_best.feature_importances_)[::-1]
columns = bank_df.columns.drop('y')

print(columns[ind][:5])
print(clf_best.feature_importances_[ind][:5])
