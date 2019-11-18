import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_path = 'data/bank.csv'

bank_df = pd.read_csv(data_path,sep=',')
print("\nbank_df.head()=\n{}".format(bank_df.head()))

#-----------------------
#練習問題1
print("\nbank_df.tail(10)=\n{}".format(bank_df.tail(10)))
#-----------------------

print("\nbank_df.shape=\n{}".format(bank_df.shape))
print("\nbank_df.dtypes=\n{}".format(bank_df.dtypes))

#欠損値の確認
print("\nbank_df.isnull().any(axis=1)=\n{}".format(bank_df.isnull().any(axis=1)))
print("\nbank_df.isnull().any(axis=0)=\n{}".format(bank_df.isnull().any(axis=0)))
#欠損値の個数
print("\nbank_df.isnull().sum(axis=1)=\n{}".format(bank_df.isnull().sum(axis=1)))
print("\nbank_df.isnull().sum(axis=0)=\n{}".format(bank_df.isnull().sum(axis=0)))

#-----------------------
#練習問題2
#print(bank_df.isnull().sum(axis=1).sort_values(ascending=False))
bank_np = bank_df.to_numpy()
bank_np = bank_np[bank_df.isnull().sum(axis=1).sort_values(ascending=False).index]
bank_df_nan = pd.DataFrame(bank_np)
print(bank_df_nan)
#----------------------

print("\nbank_df.describe()=\n{}".format(bank_df.describe()))

#-----------------------
#練習問題3
print("\nbank_df.describe(include='all')=\n{}".format(bank_df.describe(include='all')))
#-----------------------

plt.hist(bank_df['age'])
plt.xlabel('age')
plt.ylabel('freq')

plt.show()

#-----------------------
#練習問題4
plt.hist(bank_df['balance'])
plt.xlabel('balance')
plt.ylabel('freq')
plt.show()

plt.hist(bank_df['day'])
plt.xlabel('day')
plt.ylabel('freq')
plt.show()

#plt.hist(bank_df['month'])
#plt.xlabel('month')
#plt.ylabel('freq')
#plt.show()

plt.hist(bank_df['duration'])
plt.xlabel('duration')
plt.ylabel('freq')
plt.show()

plt.hist(bank_df['campaign'])
plt.xlabel('campaign')
plt.ylabel('freq')
plt.show()

plt.hist(bank_df['pdays'])
plt.xlabel('pdays')
plt.ylabel('freq')
plt.show()

plt.hist(bank_df['previous'])
plt.xlabel('previous')
plt.ylabel('freq')
plt.show()
#-----------------------

print("\nbank_df[['age','balance']].corr()=\n{}".format(bank_df[['age','balance']].corr()))
plt.scatter(bank_df['age'],bank_df['balance'])
plt.xlabel('age')
plt.ylabel('balance')
plt.show()

#-----------------------
#練習問題5
sns.pairplot(bank_df[['age','balance','duration']])
plt.show()
#-----------------------

print("\nbank_df['job'].value_counts(ascending=False,normalize=True)=\n{}".format(bank_df['job'].value_counts(ascending=False,normalize=True)))

job_label = bank_df['job'].value_counts(ascending=False,normalize=True).index
job_vals = bank_df['job'].value_counts(ascending=False,normalize=True).values

plt.pie(job_vals,labels=job_label)
plt.axis('equal')
plt.show()

#-----------------------
#練習問題6
label = bank_df['marital'].value_counts(ascending=False,normalize=True).index
vals = bank_df['marital'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['education'].value_counts(ascending=False,normalize=True).index
vals = bank_df['education'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['default'].value_counts(ascending=False,normalize=True).index
vals = bank_df['default'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['housing'].value_counts(ascending=False,normalize=True).index
vals = bank_df['housing'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['loan'].value_counts(ascending=False,normalize=True).index
vals = bank_df['loan'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['contact'].value_counts(ascending=False,normalize=True).index
vals = bank_df['contact'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['month'].value_counts(ascending=False,normalize=True).index
vals = bank_df['month'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()

label = bank_df['poutcome'].value_counts(ascending=False,normalize=True).index
vals = bank_df['poutcome'].value_counts(ascending=False,normalize=True).values
plt.pie(vals,labels=label)
plt.axis('equal')
plt.show()
#-----------------------

y_label = bank_df['y'].value_counts(ascending=False,normalize=True).index
y_vals = bank_df['y'].value_counts(ascending=False,normalize=True).values
plt.pie(y_vals,labels=y_label)
plt.axis('equal')
plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_age = [y_yes['age'],y_no['age']]

plt.boxplot(y_age)
plt.xlabel('y')
plt.xlabel('age')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()


#-----------------------
#練習問題7
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['balance'],y_no['balance']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('balance')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['day'],y_no['day']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('day')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()

#y_yes = bank_df[bank_df['y'] == 'yes']
#y_no = bank_df[bank_df['y'] == 'no']
#y_val = [y_yes['month'],y_no['month']]
#plt.boxplot(y_val)
#plt.xlabel('y')
#plt.xlabel('month')
#ax = plt.gca()
#plt.setp(ax,xticklabels=['yes','no'])
#plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['duration'],y_no['duration']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('duration')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['campaign'],y_no['campaign']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('campaign')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['pdays'],y_no['pdays']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('pdays')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()

y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
y_val = [y_yes['previous'],y_no['previous']]
plt.boxplot(y_val)
plt.xlabel('y')
plt.xlabel('previous')
ax = plt.gca()
plt.setp(ax,xticklabels=['yes','no'])
plt.show()
#-----------------------


