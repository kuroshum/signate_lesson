import matplotlib.pyplot as plt
import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')

# bank_dfのageの項目のヒストグラムを作成
plt.hist(bank_df['age'])
# x,y軸のラベル名を指定
plt.xlabel('age')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのbalanceの項目のヒストグラムを作成
plt.hist(bank_df['balance'])
# x,y軸のラベル名を指定
plt.xlabel('balance')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのdayの項目のヒストグラムを作成
plt.hist(bank_df['day'])
# x,y軸のラベル名を指定
plt.xlabel('day')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのdurationの項目のヒストグラムを作成
plt.hist(bank_df['duration'])
# x,y軸のラベル名を指定
plt.xlabel('duration')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのcampaignの項目のヒストグラムを作成
plt.hist(bank_df['campaign'])
# x,y軸のラベル名を指定
plt.xlabel('campaign')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのcampaignの項目のヒストグラムを作成
plt.hist(bank_df['pdays'])
# x,y軸のラベル名を指定
plt.xlabel('pdays')
plt.ylabel('freq')
# 表示
plt.show()

# bank_dfのpreviousの項目のヒストグラムを作成
plt.hist(bank_df['previous'])
# x,y軸のラベル名を指定
plt.xlabel('previous')
plt.ylabel('freq')
# 表示
plt.show()
