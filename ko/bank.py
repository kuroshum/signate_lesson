import pandas as pd
import matplotlib.pyplot as plt
import pdb

data_path = 'data/bank.csv'

bank_df = pd.read_csv(data_path, sep=',')
print(bank_df.head())
print(bank_df.tail(10)) # 練習問題1の回答

print(bank_df.shape)
print(bank_df.dtypes)

# 欠損値の確認
print(bank_df.isnull().any(axis=1))
print(bank_df.isnull().sum(axis=1))
print(bank_df.isnull().any(axis=0))
print(bank_df.isnull().sum(axis=0))

#---------- 練習問題2の回答 ----------
print(bank_df.isnull().sum(axis=1).sort_values(ascending=False))
print(bank_df.isnull().sum(axis=0).sort_values(ascending=False))
#----------------------------------

# 統計量の計算
print(bank_df.describe())
print(bank_df.describe(include = object)) # 練習問題3の回答

# データの可視化
# bank_dfのage項目のヒストグラムを作成
plt.hist(bank_df['age'])
# x,y軸のラベル名を指定
plt.xlabel('age')
plt.ylabel('freq')
# 表示
plt.show()

#---------- 練習問題4の回答 ----------
# bank_dfのjob項目のヒストグラムを作成
plt.hist(bank_df['balance'])
# x,y軸のラベル名を指定
plt.xlabel('balance')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのmartial項目のヒストグラムを作成
plt.hist(bank_df['day'])
# x,y軸のラベル名を指定
plt.xlabel('day')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのduration項目のヒストグラムを作成
plt.hist(bank_df['duration'])
# x,y軸のラベル名を指定
plt.xlabel('duration')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのcampaign項目のヒストグラムを作成
plt.hist(bank_df['campaign'])
# x,y軸のラベル名を指定
plt.xlabel('campaign')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのpdays項目のヒストグラムを作成
plt.hist(bank_df['pdays'])
# x,y軸のラベル名を指定
plt.xlabel('pdays')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのprevious項目のヒストグラムを作成
plt.hist(bank_df['previous'])
# x,y軸のラベル名を指定
plt.xlabel('previous')
plt.ylabel('freq')
# 表示
plt.show()
#----------------------------------

print(bank_df[['age','balance']].corr())
plt.scatter(bank_df['age'], bank_df['balance'])
plt.xlabel('age')
plt.ylabel('balance')
plt.show()

#---------- 練習問題5の回答 ----------
print(bank_df[['age','duration']].corr())
plt.scatter(bank_df['age'], bank_df['duration'])
plt.xlabel('age')
plt.ylabel('duration')
plt.show()

print(bank_df[['balance','duration']].corr())
plt.scatter(bank_df['balance'], bank_df['duration'])
plt.xlabel('balance')
plt.ylabel('duration')
plt.show()
#----------------------------------

print(bank_df['job'].value_counts(ascending=False, normalize=True))
job_label = bank_df['job'].value_counts(ascending=False, normalize=True).index
job_vals = bank_df['job'].value_counts(ascending=False, normalize=True).values
# 円グラフ
plt.pie(job_vals, labels=job_label)
plt.axis('equal')
plt.show()

#---------- 練習問題6の回答 ----------
marital_label = bank_df['marital'].value_counts(ascending=False, normalize=True).index
marital_vals = bank_df['marital'].value_counts(ascending=False, normalize=True).values
plt.pie(marital_vals, labels=marital_label)
plt.axis('equal')
plt.show()

education_label = bank_df['education'].value_counts(ascending=False, normalize=True).index
education_vals = bank_df['education'].value_counts(ascending=False, normalize=True).values
plt.pie(education_vals, labels=education_label)
plt.axis('equal')
plt.show()

default_label = bank_df['default'].value_counts(ascending=False, normalize=True).index
default_vals = bank_df['default'].value_counts(ascending=False, normalize=True).values
plt.pie(default_vals, labels=default_label)
plt.axis('equal')
plt.show()

housing_label = bank_df['housing'].value_counts(ascending=False, normalize=True).index
housing_vals = bank_df['housing'].value_counts(ascending=False, normalize=True).values
plt.pie(housing_vals, labels=housing_label)
plt.axis('equal')
plt.show()

loan_label = bank_df['loan'].value_counts(ascending=False, normalize=True).index
loan_vals = bank_df['loan'].value_counts(ascending=False, normalize=True).values
plt.pie(loan_vals, labels=loan_label)
plt.axis('equal')
plt.show()

contact_label = bank_df['contact'].value_counts(ascending=False, normalize=True).index
contact_vals = bank_df['contact'].value_counts(ascending=False, normalize=True).values
plt.pie(contact_vals, labels=contact_label)
plt.axis('equal')
plt.show()

month_label = bank_df['month'].value_counts(ascending=False, normalize=True).index
month_vals = bank_df['month'].value_counts(ascending=False, normalize=True).values
plt.pie(month_vals, labels=month_label)
plt.axis('equal')
plt.show()

poutcome_label = bank_df['poutcome'].value_counts(ascending=False, normalize=True).index
poutcome_vals = bank_df['poutcome'].value_counts(ascending=False, normalize=True).values
plt.pie(poutcome_vals, labels=poutcome_label)
plt.axis('equal')
plt.show()

y_label = bank_df['y'].value_counts(ascending=False, normalize=True).index
y_vals = bank_df['y'].value_counts(ascending=False, normalize=True).values
plt.pie(y_vals, labels=y_label)
plt.axis('equal')
plt.show()
#----------------------------------

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータそれぞれの年齢を格納
y_age = [y_yes['age'], y_no['age']]
# 箱ひげ図
plt.boxplot(y_age)
plt.xlabel('y')
plt.ylabel('age')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

#---------- 練習問題7の回答 ----------
y_balance = [y_yes['balance'], y_no['balance']]
plt.boxplot(y_balance)
plt.xlabel('y')
plt.ylabel('balance')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

y_day = [y_yes['day'], y_no['day']]
plt.boxplot(y_day)
plt.xlabel('y')
plt.ylabel('day')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

y_duration = [y_yes['duration'], y_no['duration']]
plt.boxplot(y_duration)
plt.xlabel('y')
plt.ylabel('duration')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

y_campaign = [y_yes['campaign'], y_no['campaign']]
plt.boxplot(y_campaign)
plt.xlabel('y')
plt.ylabel('campaign')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

y_pdays = [y_yes['pdays'], y_no['pdays']]
plt.boxplot(y_pdays)
plt.xlabel('y')
plt.ylabel('pdays')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

y_previous = [y_yes['previous'], y_no['previous']]
plt.boxplot(y_previous)
plt.xlabel('y')
plt.ylabel('previous')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()
#----------------------------------

# 欠損値の除外
bank_df = bank_df.dropna(subset = ['job','education'])
print(bank_df.shape)
bank_df = bank_df.drop(['poutcome'], axis = 1) # 練習問題8の回答
print(bank_df.shape)
# 欠損値の補完
bank_df = bank_df.fillna({'contact':'unknown'})
print(bank_df.head())
# 外れ値の除外
bank_df = bank_df[bank_df['age'] >= 18]
bank_df = bank_df[bank_df['age'] < 100]
print(bank_df.shape)
# 文字列（二値データ）を数値に変換
bank_df = bank_df.replace('yes',1)
bank_df = bank_df.replace('no',0)
print(bank_df.head())
# 文字列（多値データ）を数値に変換
bank_df_job = pd.get_dummies(bank_df['job'])
print(bank_df_job.head())
print(bank_df.dtypes)
# ---------- 練習問題9の回答 ----------
bank_df_marital = pd.get_dummies(bank_df['marital'])
print(bank_df_marital.head())
bank_df_education = pd.get_dummies(bank_df['education'])
print(bank_df_education.head())
bank_df_contact = pd.get_dummies(bank_df['contact'])
print(bank_df_contact.head())
bank_df_month = pd.get_dummies(bank_df['month'])
print(bank_df_month.head())
# ----------------------------------

# 分析データセットの作成
tmp0 = bank_df[['age', 'default', 'balance', 'housing', 'loan', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']]
print(tmp1.head())
tmp1 = pd.concat([tmp0, bank_df_job], axis=1)
tmp2 = pd.concat([tmp1, bank_df_marital], axis=1)
tmp3 = pd.concat([tmp2, bank_df_education], axis=1)
tmp4 = pd.concat([tmp3, bank_df_contact], axis=1)
bank_df_new = pd.concat([tmp4, bank_df_month], axis=1)
print(bank_df_new.head())
print(bank_df_new.dtypes)
