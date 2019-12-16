import matplotlib.pyplot as plt
import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')

print(bank_df[['age','balance']].corr())

plt.scatter(bank_df['age'], bank_df['balance'])
plt.xlabel('age')
plt.ylabel('balance')
plt.show()


print(bank_df[['age','duration']].corr())

plt.scatter(bank_df['age'], bank_df['duration'])
plt.xlabel('age')
plt.ylabel('duration')
plt.show()



print(bank_df[['duration','balance']].corr())

plt.scatter(bank_df['duration'], bank_df['balance'])
plt.xlabel('duration')
plt.ylabel('balance')
plt.show()