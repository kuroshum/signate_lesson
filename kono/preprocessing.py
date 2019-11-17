import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	bank_df = pd.read_csv('bank.csv', sep=',')
	
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
