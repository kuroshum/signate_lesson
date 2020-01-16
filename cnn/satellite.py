import pandas as pd
import pickle as pkl
import os
import sys
import time

if __name__ == '__main__':
	stop = 48
	step = 6
	start = time.time()
	diff = []
	for year in range(1995, 2017 + 1, 1):
		with open('/home/kurora/satellite_data/satellite_fill' + str(year) + '.pkl', 'rb') as file:
			df = pkl.load(file)
		lists = []
		for name, df0 in df.sort_values('ID').groupby('ID'):
			# 'ID'と'NUM'が連動していない場合は例外を投げる
			if not df0[df0['NUM'] != df0['NUM'][0]].empty:
				raise Exception('Error')
			df1 = df0[df0['type'] == 'IR_ty'].sort_index()
			df2 = df0[df0['type'] == 'WV_ty'].sort_index()
			df3 = df0[df0['type'] == 'diff_ty'].sort_index()
			df4 = df0[df0['type'] == 'IR_wide'].sort_index()
			values = pd.DataFrame(columns=['date', 'lon', 'lat', 'wide', 'ty', 'WV', 'Diff', 'cp', 'true', 'NUM', 'ID'])
			i = df1.index[0]
			while i <= df1.index[len(df1.index) - 1] - pd.offsets.Hour(stop):
				series = [[], [], [], [], [], [], [], [], [], [], []]
				j = i
				while j <= i + pd.offsets.Hour(stop):
					# 任意の時刻におけるデータが存在しない場合は追加をしない
					if df1[df1.index == j].empty or df2[df2.index == j].empty or df3[df3.index == j].empty or df4[df4.index == j].empty:
						print('Missing: {0}'.format(j), file=sys.stderr)
						j += pd.offsets.Hour(step)
						continue
					# 任意の時刻におけるデータが複数存在する場合は例外を投げる
					if len(df1[df1.index == j]) > 1 or len(df2[df2.index == j]) > 1 or len(df3[df3.index == j]) > 1 or len(df4[df4.index == j]) > 1:
						raise Exception('Error')
					# 'type'ごとに'lon'が異なる場合は例外を投げる
					if df1[df1.index == j]['lon'][0] != df2[df2.index == j]['lon'][0] or df1[df1.index == j]['lon'][0] != df3[df3.index == j]['lon'][0] or df1[df1.index == j]['lon'][0] != df4[df4.index == j]['lon'][0]:
						raise Exception('Error')
					# 'type'ごとに'lat'が異なる場合は例外を投げる
					if df1[df1.index == j]['lat'][0] != df2[df2.index == j]['lat'][0] or df1[df1.index == j]['lat'][0] != df3[df3.index == j]['lat'][0] or df1[df1.index == j]['lat'][0] != df4[df4.index == j]['lat'][0]:
						raise Exception('Error')
					# 'type'ごとに'cp'が異なる場合は例外を投げる
					if df1[df1.index == j]['cp'][0] != df2[df2.index == j]['cp'][0] or df1[df1.index == j]['cp'][0] != df3[df3.index == j]['cp'][0] or df1[df1.index == j]['cp'][0] != df4[df4.index == j]['cp'][0]:
						raise Exception('Error')
					series[0].append(df1[df1.index == j].index)
					series[1].append(df1[df1.index == j]['lon'][0])
					series[2].append(df1[df1.index == j]['lat'][0])
					series[3].append(df4[df4.index == j]['img'].values[0])
					series[4].append(df1[df1.index == j]['img'].values[0])
					series[5].append(df2[df2.index == j]['img'].values[0])
					series[6].append(df3[df3.index == j]['img'].values[0])
					series[7].append(df1[df1.index == j]['cp'][0])
					series[8].append(False)
					series[9].append(df1[df1.index == j]['NUM'][0])
					series[10].append(df1[df1.index == j]['ID'][0])
					j += pd.offsets.Hour(step)
				# 単位時間あたりの中心気圧の低下量を計算する
				diff.append((series[7][len(series[7]) - 1] - series[7][0]) / ((series[0][len(series[0]) - 1] - series[0][0]) / pd.Timedelta('1 hour'))[0])
				values = values.append(pd.Series(series, index=values.columns), ignore_index=True)
				i += pd.offsets.Hour(step)
			lists.append(values)
		with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satellite_window/satellite_window_fill' + str(year) + '.pkl'), 'wb') as file:
			pkl.dump(lists, file, -1)
	diff.sort()
	threshold = diff[int(len(diff) * 0.05)]
	print('Threshold: {0}'.format(threshold), file=sys.stdout)
	for year in range(1995, 2017 + 1, 1):
		with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satellite_window/satellite_window_fill' + str(year) + '.pkl'), 'rb') as file:
			lists = pkl.load(file)
		for df in lists:
			for index, series in df.iterrows():
				if (series['cp'][len(series['cp']) - 1] - series['cp'][0]) / ((series['date'][len(series['date']) - 1] - series['date'][0]) / pd.Timedelta('1 hour'))[0] <= threshold:
					series['true'] = [True] * len(series['true'])
		with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satellite_window/satellite_window_fill' + str(year) + '.pkl'), 'wb') as file:
			pkl.dump(lists, file, -1)
	print('Time: {0}'.format(time.time() - start), file=sys.stdout)
