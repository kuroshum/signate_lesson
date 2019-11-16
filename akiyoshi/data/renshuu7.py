import matplotlib.pyplot as plt
import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_age = [y_yes['age'],y_no['age']]

# 箱ひげ図
plt.boxplot(y_age)
plt.xlabel('y')
plt.ylabel('age')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_balance = [y_yes['balance'],y_no['balance']]

# 箱ひげ図
plt.boxplot(y_balance)
plt.xlabel('y')
plt.ylabel('balance')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_day = [y_yes['day'],y_no['day']]

# 箱ひげ図
plt.boxplot(y_day)
plt.xlabel('y')
plt.ylabel('day')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_duration = [y_yes['duration'],y_no['duration']]

# 箱ひげ図
plt.boxplot(y_duration)
plt.xlabel('y')
plt.ylabel('duration')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_camption = [y_yes['campaign'],y_no['campaign']]

# 箱ひげ図
plt.boxplot(y_camption)
plt.xlabel('y')
plt.ylabel('campaign')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_pdays = [y_yes['pdays'],y_no['pdays']]

# 箱ひげ図
plt.boxplot(y_pdays)
plt.xlabel('y')
plt.ylabel('pdays')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()

# yes, noのデータをそれぞれ格納
y_yes = bank_df[bank_df['y'] == 'yes']
y_no = bank_df[bank_df['y'] == 'no']
# yes, noのデータのそれぞれ年齢を格納
y_previous = [y_yes['previous'],y_no['previous']]

# 箱ひげ図
plt.boxplot(y_previous)
plt.xlabel('y')
plt.ylabel('previous')
ax = plt.gca()
plt.setp(ax, xticklabels=['yes','no'])
plt.show()