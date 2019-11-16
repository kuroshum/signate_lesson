import numpy as np

values = np.array([10, 3, 1, 5, 8, 6])

convert_values = np.array([])

#for ind in np.arange(len(values)):
#    if values[ind] >= 5:
#        convert_values = np.append(convert_values,1)
#    elif values[ind] < 5:
#        convert_values = np.append(convert_values,-1)

convert_values = [1 if values[ind] >= 5 else -1 for ind in np.arange(len(values))]

print(f"convert_values:\n{convert_values}")