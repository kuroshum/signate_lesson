import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')
print(bank_df.describe(include='all'))

