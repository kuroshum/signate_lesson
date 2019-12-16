import pandas as pd

data_path = 'data/bank.csv'

bank_df = pd.read_csv(data_path,sep=',')
print("\nbank_df.head()=\n{}".format(bank_df.head()))

print("\nbank_df.shape=\n{}".format(bank_df.shape))
print("\nbank_df.dtypes=\n{}".format(bank_df.dtypes))

bank_df = bank_df.dropna(subset=['job','education'])
print("\nbank_df.dropna(subset=['job','education'].shape=\n{}".format(bank_df.shape))

#-----------------------
#練習問題8
bank_df = bank_df.drop('poutcome',axis=1)
print("\nbank_df.drop('poutcome',axis=1).shape=\n{}".format(bank_df.shape))
#-----------------------
print("\nbank_df.dtypes=\n{}".format(bank_df.dtypes))


bank_df = bank_df.fillna({'contact':'unknown'})
print("\nbank_df.fillna()=\n{}".format(bank_df.head()))

bank_df = bank_df[bank_df['age']>=18]
bank_df = bank_df[bank_df['age']<100]
print("\nbank_df.shape=\n{}".format(bank_df.shape))

bank_df = bank_df.replace('yes',1)
bank_df = bank_df.replace('no',0)
print("\nbank_df.head()=\n{}".format(bank_df.head()))

bank_df_job = pd.get_dummies(bank_df['job'])
print("\nbank_df_job.head()=\n{}".format(bank_df_job.head()))

#-----------------------
#練習問題9
bank_df_marital = pd.get_dummies(bank_df['marital'])
bank_df_education = pd.get_dummies(bank_df['education'])
bank_df_contact = pd.get_dummies(bank_df['contact'])
bank_df_month = pd.get_dummies(bank_df['month'])
#-----------------------

tmp1 = bank_df[['age','default','balance','housing','loan','day','duration','campaign','pdays','previous','y']]
print("\ntmp1.head()=\n{}".format(tmp1.head()))

tmp2 = pd.concat([tmp1,bank_df_marital],axis=1)
tmp3 = pd.concat([tmp2,bank_df_education],axis=1)
tmp4 = pd.concat([tmp3,bank_df_contact],axis=1)
bank_df_new = pd.concat([tmp4,bank_df_month],axis=1)
print("\nbank_df_new.head()=\n{}".format(bank_df_new.head()))

bank_df_new.to_csv('bank_prep.csv',index=False)

