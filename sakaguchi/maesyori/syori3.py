import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

bank_df = pd.read_csv('data/bank-prep.csv',sep=',')
print("\nbank_df.head()=\n{}".format(bank_df.head()))

X = np.array(bank_df.drop('y',axis=1))
Y = np.array(bank_df[['y']])
print(np.sum(Y==1),np.sum(Y==0))

sampler = RandomUnderSampler(random_state=42)
X,Y = sampler.fit_resample(X,Y)
print(np.sum(Y==1),np.sum(Y==0))

from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=18,shuffle=True)
scores = []

for train_id,test_id in kf.split(X):
    x = X[train_id]
    y = Y[train_id]

    cif = tree.DecisionTreeClassifier()
    cif.fit(x,y)
    pred_y = cif.predict(X[test_id])
    score = accuracy_score(Y[test_id],pred_y)
    scores.append(score)

scores = np.array(scores)
print(scores.mean(),scores.std())


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

print(recall_score(Y[test_id],pred_y))
print(precision_score(Y[test_id],pred_y))

print(cif)

from sklearn.model_selection import GridSearchCV

params = {'criterion':['entropy'],'max_depth':[2,4,6,8,10],'min_samples_leaf':[10,20,30,40,50]}

cif_gs = GridSearchCV(tree.DecisionTreeClassifier(),params,cv=KFold(n_splits=10,shuffle=True),scoring='accuracy')

cif_gs.fit(X,Y)

print(cif_gs.best_score_)
print(cif_gs.best_params_)

cif_best = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_leaf=20)
cif_best.fit(X,Y)

print(cif_best.feature_importances_)

ind = np.argsort(cif_best.feature_importances_)[::-1]
colums = bank_df.columns.drop('y')

print(colums[ind][:5])
print(cif_best.feature_importances_[ind][:5])

