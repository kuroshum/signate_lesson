import numpy as np

values = np.array([10, 3, 1, 5, 8, 6])

#---------------
# 通常のfor文
# 空のarray
passed_values = np.array([])
for ind in np.arange(len(values)):
	#通常のif文
	if values[ind] > 5:
		passed_values = np.append(passed_values,values[ind])

# 結果を標準出力
passed_values = passed_values.astype('int') #int型にキャスト
print("5以上の値",passed_values)
#---------------

#---------------
# リスト内包表記のfor文
passed_values = values[[values[ind] > 5 for ind in np.arange(len(values))]]

# 結果を標準出力
print("5以上の値",passed_values)
#---------------
