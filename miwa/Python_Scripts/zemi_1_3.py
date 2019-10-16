#演習３
import numpy as np
values = np.array([10, 3, 1, 5, 8, 6])
a=np.array([[1],[-1]])
convert_values = np.array([])
for ind in np.arange(len(values)):
    if values[ind] >= 5:
        convert_values=np.append(convert_values,a[0])
    else:
        convert_values=np.append(convert_values,a[-1])


print(convert_values)
