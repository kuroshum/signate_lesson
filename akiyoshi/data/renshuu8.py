import matplotlib.pyplot as plt
import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')

bank_df = bank_df.dropna(subset=['job', 'education'])
print(bank_df.shape)

bank_df = bank_df.drop(['poutcome'], axis = 1)
print(bank_df.shape)

bank_df = bank_df.fillna({'contact': 'unknown'})
print(bank_df.head())

bank_df = bank_df[bank_df['age'] >= 18]
bank_df = bank_df[bank_df['age'] < 100]
print(bank_df.shape)

bank_df = bank_df.replace('yes',1)
bank_df = bank_df.replace('no',0)

print(bank_df.head())

bank_df_job = pd.get_dummies(bank_df['job'])
print(bank_df_job.head())

bank_df_education = pd.get_dummies(bank_df['education'])
print(bank_df_job.head())

bank_df_marital = pd.get_dummies(bank_df['marital'])
print(bank_df_job.head())

bank_df_contact = pd.get_dummies(bank_df['contact'])
print(bank_df_job.head())

bank_df_month = pd.get_dummies(bank_df['month'])
print(bank_df_job.head())

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

bank_df_new.to_csv('bank_df_new.csv')


