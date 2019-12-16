import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')
#print(bank_df.isnull().any(axis=1))
#print(bank_df.isnull().any(axis=0))
#print(bank_df.isnull().sum(axis=1))
#print(bank_df.isnull().sum(axis=0))


print(bank_df.isnull().sum(axis=0).sort_values(ascending = False))
print(bank_df.isnull().sum(axis=1).sort_values(ascending = False))