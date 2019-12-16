import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_path = 'data/bank.csv'

bank_df = pd.read_csv(data_path, sep=',')
#Exercise1
"""print(bank_df.tail(10))
print(bank_df.shape)
print(bank_df.dtypes)"""
#Exercise2
"""print(bank_df.isnull().any(axis=1))
print(bank_df.isnull().any(axis=0))
print(bank_df.isnull().sum(axis=1).sort_values(ascending=False))
print(bank_df.isnull().sum(axis=0).sort_values(ascending=False))"""

#Exercise3
#print(bank_df.describe(include='all'))


#Exercise4
"""plt.hist(bank_df['age'])
plt.xlabel('age')
plt.ylabel('freq')
plt.savefig('bank_age.png')

plt.hist(bank_df['balance'])
plt.xlabel('balance')
plt.ylabel('freq')
plt.savefig('bank_balance.png')

plt.hist(bank_df['day'])
plt.xlabel('day')
plt.ylabel('freq')
plt.savefig('bank_day.png')

plt.hist(bank_df['duration'])
plt.xlabel('duration')
plt.ylabel('freq')
plt.savefig('bank_duration.png')

plt.hist(bank_df['compaogn'])
plt.xlabel('compaign')
plt.ylabel('freq')
plt.savefig('bank_compaign.png')

plt.hist(bank_df['pdays'])
plt.xlabel('pdays')
plt.ylabel('freq')
plt.savefig('bank_pdays.png')

plt.hist(bank_df['previous'])
plt.xlabel('previous')
plt.ylabel('freq')
plt.savefig('bank_previous.png')"""


#Exercise5
"""print(bank_df[['age','balance','duration']].corr())
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(bank_df['age'], bank_df['balance'],bank_df['duration'])

ax.set_xlabel('age')
ax.set_ylabel('balance')
ax.set_zlabel('duration')
plt.savefig('bank_age_balamce_scatter')"""
"""print(bank_df['job'].value_counts(ascending=False, normalize=True))
job_label = bank_df['job'].value_counts(ascending=False, normalize=True).index
job_vals = bank_df['job'].value_counts(ascending=False, normalize=True).values

plt.pie(job_vals, labels=job_label)
plt.axis('equal')
plt.savefig('bank_job_circle')"""

#Exercise6
"""y_label = bank_df['y'].value_counts(ascending=False, normalize=True).index
y_vals = bank_df['y'].value_counts(ascending=False, normalize=True).values
plt.pie(y_vals, labels=y_label)
plt.axis('equal')
plt.savefig('bank_y_circle')"""

#Exercise7
# yes, noのデータをそれぞれ格納
"""y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_age = [y_yes['age'],y_no['age']]

# 箱ひげ図
plt.boxplot(y_age)
plt.xlabel('y')
plt.ylabel('age')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.savefig('bank_y_age_box')"""

#Exercise8
"""print(bank_df.shape)
bank_df = bank_df.dropna(subset=['job', 'education'])
print(bank_df.shape)
bank_df = bank_df.dropna(subset=['poutcome'])
print(bank_df.shape)"""

#Exercise9
bank_df = bank_df.fillna({'contact': 'unknown'})
print(bank_df.head())
bank_df = bank_df[bank_df['age'] >= 18]
bank_df = bank_df[bank_df['age'] < 100]
print(bank_df.shape)
bank_df = bank_df.replace('yes',1)
bank_df = bank_df.replace('no',0)
print(bank_df.head())
bank_df_job = pd.get_dummies(bank_df['job'])
bank_df_marital = pd.get_dummies(bank_df['marital'])
bank_df_education = pd.get_dummies(bank_df['education'])
bank_df_default = pd.get_dummies(bank_df['default'])
bank_df_housing = pd.get_dummies(bank_df['housing'])
bank_df_loan = pd.get_dummies(bank_df['loan'])
bank_df_contact = pd.get_dummies(bank_df['contact'])
bank_df_month = pd.get_dummies(bank_df['month'])
bank_df_poutcome = pd.get_dummies(bank_df['poutcome'])
bank_df_y = pd.get_dummies(bank_df['y'])

tmp1 = bank_df[[
    'age', 'default', 'balance', 'housing', 'loan',
    'day', 'duration', 'campaign', 'pdays', 'previous', 'y'
]]
print(tmp1.head())
tmp2 = pd.concat([tmp1, bank_df_marital], axis=1)
tmp3 = pd.concat([tmp2, bank_df_education], axis=1)
tmp4 = pd.concat([tmp3, bank_df_contact], axis=1)
bank_df_new = pd.concat([tmp4, bank_df_month], axis=1)
print(bank_df_new.head())

bank_df_new.to_csv("bank_prep.csv",index=False)
