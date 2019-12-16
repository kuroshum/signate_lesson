import matplotlib.pyplot as plt
import pandas as pd

data_path = 'bank.csv'

bank_df = pd.read_csv(data_path,sep=',')

print(bank_df['job'].value_counts(ascending=False, normalize=True))

# 値ら別と値の出現数を計算
job_label = bank_df['job'].value_counts(ascending=False, normalize=True).index
job_vals = bank_df['job'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(job_vals, labels=job_label)
plt.axis('equal')
plt.show()

marital_label = bank_df['marital'].value_counts(ascending=False, normalize=True).index
marital_vals = bank_df['marital'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(marital_vals, labels=marital_label)
plt.axis('equal')
plt.show()

education_label = bank_df['education'].value_counts(ascending=False, normalize=True).index
education_vals = bank_df['education'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(education_vals, labels=education_label)
plt.axis('equal')
plt.show()

default_label = bank_df['default'].value_counts(ascending=False, normalize=True).index
default_vals = bank_df['default'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(default_vals, labels=default_label)
plt.axis('equal')
plt.show()

housing_label = bank_df['housing'].value_counts(ascending=False, normalize=True).index
housing_vals = bank_df['housing'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(housing_vals, labels=housing_label)
plt.axis('equal')
plt.show()

loan_label = bank_df['loan'].value_counts(ascending=False, normalize=True).index
loan_vals = bank_df['loan'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(loan_vals, labels=loan_label)
plt.axis('equal')
plt.show()


contact_label = bank_df['contact'].value_counts(ascending=False, normalize=True).index
contact_vals = bank_df['contact'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(contact_vals, labels=contact_label)
plt.axis('equal')
plt.show()


month_label = bank_df['month'].value_counts(ascending=False, normalize=True).index
month_vals = bank_df['month'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(month_vals, labels=month_label)
plt.axis('equal')
plt.show()

poutcome_label = bank_df['poutcome'].value_counts(ascending=False, normalize=True).index
poutcome_vals = bank_df['poutcome'].value_counts(ascending=False, normalize=True).values

# 円グラフ
plt.pie(poutcome_vals, labels=poutcome_label)
plt.axis('equal')
plt.show()