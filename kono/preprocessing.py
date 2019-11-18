import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	bank_df = pd.read_csv('data/bank.csv', sep=',')
	
	# 練習問題1
	print(bank_df.tail(10))
	
	# 練習問題2
	print(bank_df.isnull().sum(axis=1).sort_values(ascending=False))
	print(bank_df.isnull().sum(axis=0).sort_values(ascending=False))
	
	# 練習問題3
	print(bank_df.describe(include=[object]))
	
	# 練習問題4
	for index, values in bank_df.select_dtypes(include='int64').iteritems():
		plt.hist(values)
		plt.xlabel(index)
		plt.ylabel('freq')
		plt.show()
	
	# 練習問題5
	pd.plotting.scatter_matrix(bank_df[['age', 'balance', 'duration']])
	plt.tight_layout()
	plt.show()
	
	# 練習問題6
	for index, values in bank_df.select_dtypes(include='object').iteritems():
		print(values.value_counts(ascending=False, normalize=True))
		marital_label = values.value_counts(ascending=False, normalize=True).index
		marital_vals = values.value_counts(ascending=False, normalize=True).values
		plt.pie(marital_vals, labels=marital_label)
		plt.axis('equal')
		plt.show()
	
	# 練習問題7
	y_yes = bank_df[bank_df['y'] == 'yes']
	y_no = bank_df[bank_df['y'] == 'no']
	for index, values in bank_df.select_dtypes(include='int64').iteritems():
		plt.boxplot([y_yes[index], y_no[index]])
		plt.xlabel('y')
		plt.ylabel(index)
		plt.setp(plt.gca(), xticklabels = ['yes', 'no'])
		plt.show()
	
	# 練習問題8
	bank_df = bank_df.dropna(subset=['job', 'education'])
	bank_df = bank_df.drop('poutcome', axis=1)
	print(bank_df.shape)
	
	# 練習問題9
	for index, values in bank_df.select_dtypes(include='object').drop('job', axis=1).iteritems():
		print(pd.get_dummies(values).head())
	
	bank_df = bank_df.dropna(subset=['job', 'education'])
	bank_df = bank_df.dropna(thresh=2400, axis=1)
	bank_df = bank_df.fillna({'contact':'unknown'})
	bank_df = bank_df[bank_df['age'] >= 18]
	bank_df = bank_df[bank_df['age'] < 100]
	bank_df.loc[(bank_df['job'] == 'management') | (bank_df['job'] == 'technician') | (bank_df['job'] == 'blue-collar') | (bank_df['job'] == 'admin.') | (bank_df['job'] == 'services') | (bank_df['job'] == 'self-employed') | (bank_df['job'] == 'entrepreneur') | (bank_df['job'] == 'housemaid'), 'job2'] = 'worker'
	
	# 練習問題10
	bank_df.loc[(bank_df['job'] == 'retired') | (bank_df['job'] == 'unemployed') | (bank_df['job'] == 'student'), 'job2'] = 'non-worker'
	
	# 練習問題11
	bank_df.loc[(bank_df['month'] == 'jan') | (bank_df['month'] == 'feb') | (bank_df['month'] == 'mar'), 'month2'] = '1Q'
	bank_df.loc[(bank_df['month'] == 'apr') | (bank_df['month'] == 'may') | (bank_df['month'] == 'jun'), 'month2'] = '2Q'
	bank_df.loc[(bank_df['month'] == 'jul') | (bank_df['month'] == 'aug') | (bank_df['month'] == 'sep'), 'month2'] = '3Q'
	bank_df.loc[(bank_df['month'] == 'oct') | (bank_df['month'] == 'nov') | (bank_df['month'] == 'dec'), 'month2'] = '4Q'
	
	bank_df.loc[bank_df['day'] <= 10, 'day2'] = 'early'
	
	# 練習問題12
	bank_df.loc[(bank_df['day'] > 10) & (bank_df['day'] <= 20), 'day2'] = 'middle'
	bank_df.loc[bank_df['day'] > 20, 'day2'] = 'late'
	
	bank_df.loc[bank_df['duration'] < 300, 'duration2'] = 'short'
	bank_df.loc[bank_df['duration'] >= 300, 'duration2'] = 'long'
	bank_df.loc[bank_df['previous'] < 1, 'previous2'] = 'zero'
	bank_df.loc[bank_df['previous'] >= 1, 'previous2'] = 'one-more'
	
	# 練習問題13
	bank_df.loc[bank_df['pdays'] < 0, 'pdays2'] = 'less'
	bank_df.loc[bank_df['pdays'] >= 0, 'pdays2'] = 'more'
	print(bank_df.head())
