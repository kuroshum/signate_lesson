import numpy as np

values = np.array([10, 3, 1, 5, 8, 6])

#---------------
# 通常のfor文
# 空のarray
convert_values = np.array([])
convert_values2 = np.array([])
for ind in np.arange(len(values)):
	if values[ind] >= 5:
		convert_values = np.append(convert_values,1)
	else:
		convert_values = np.append(convert_values,-1)

# 結果を標準出力
convert_values = convert_values.astype('int') #int型にキャスト
print("arrays:",convert_values)
#---------------

#---------------
# リスト内包表記のfor文
convert_values2 = values[[np.append(convert_values2,1) if values[ind] >= 5 else np.append(convert_values2,-1) for ind in np.arange(len(values))]]

# 結果を標準出力
print("5以上の値",convert_values2)
#---------------
