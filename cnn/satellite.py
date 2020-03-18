import pandas as pd
import pickle as pkl
import os
import sys
import time

def make_window(source, dest, year1, year2, stop=48, step=1):
	diff = []
	for year in range(year1, year2 + 1, 1):
		with open(os.path.join(source, 'satellite_fill') + str(year) + '.pkl', 'rb') as file:
			df = pkl.load(file)
		lists = []
		for name, df0 in df.sort_values('ID').groupby('ID'):
			# 'ID'と'NUM'が連動していない場合は例外を投げる
			if not df0[df0['NUM'] != df0['NUM'][0]].empty:
				raise Exception('Error')
			df0 = df0.sort_index()
			# 任意の時刻におけるデータが複数存在する場合は例外を投げる
			df1 = df0[df0['type'] == 'IR_ty'].to_dict(orient='index')
			df2 = df0[df0['type'] == 'WV_ty'].to_dict(orient='index')
			df3 = df0[df0['type'] == 'diff_ty'].to_dict(orient='index')
			df4 = df0[df0['type'] == 'IR_wide'].to_dict(orient='index')
			diff.extend(df0['cp'].values.tolist())
			values = []
			i = df0.index[0]
			while i <= df0.index[len(df0.index) - 1] - pd.offsets.Hour(stop):
				series = [[], [], [], [], [], [], [], [], [], [], []]
				j = i
				while j <= i + pd.offsets.Hour(stop):
					# 任意の時刻におけるデータが存在しない場合は追加をしない
					if j not in df1 or j not in df2 or j not in df3 or j not in df4:
						print('Missing: {0}'.format(j), file=sys.stderr)
						j += pd.offsets.Hour(step)
						continue
					# 'type'ごとに'lon'が異なる場合は例外を投げる
					if df1[j]['lon'] != df2[j]['lon'] or df1[j]['lon'] != df3[j]['lon'] or df1[j]['lon'] != df4[j]['lon']:
						raise Exception('Error')
					# 'type'ごとに'lat'が異なる場合は例外を投げる
					if df1[j]['lat'] != df2[j]['lat'] or df1[j]['lat'] != df3[j]['lat'] or df1[j]['lat'] != df4[j]['lat']:
						raise Exception('Error')
					# 'type'ごとに'cp'が異なる場合は例外を投げる
					if df1[j]['cp'] != df2[j]['cp'] or df1[j]['cp'] != df3[j]['cp'] or df1[j]['cp'] != df4[j]['cp']:
						raise Exception('Error')
					series[0].append(j)
					series[1].append(df1[j]['lon'])
					series[2].append(df1[j]['lat'])
					series[3].append(df4[j]['img'])
					series[4].append(df1[j]['img'])
					series[5].append(df2[j]['img'])
					series[6].append(df3[j]['img'])
					series[7].append(df1[j]['cp'])
					series[8].append(False)
					series[9].append(df1[j]['NUM'])
					series[10].append(df1[j]['ID'])
					j += pd.offsets.Hour(step)
				values.append(series)
				i += pd.offsets.Hour(step)
			lists.append(pd.DataFrame.from_dict(dict(zip(range(len(values)), values)), columns=['date', 'lon', 'lat', 'wide', 'ty', 'WV', 'Diff', 'cp', 'true', 'NUM', 'ID'], orient='index'))
		with open(os.path.join(dest, 'satellite_window_fill' + str(year) + '.pkl'), 'wb') as file:
			pkl.dump(lists, file, -1)
	diff.sort()
	threshold = diff[int(len(diff) * 0.05)]
	print('Threshold ({0}-{1}): {2}'.format(year1, year2, threshold), file=sys.stdout)
	for year in range(year1, year2 + 1, 1):
		with open(os.path.join(dest, 'satellite_window_fill' + str(year) + '.pkl'), 'rb') as file:
			lists = pkl.load(file)
		for df in lists:
			for index, series in df.iterrows():
				if series['cp'][len(series['cp']) - 1] <= threshold:
					series['true'] = [True] * len(series['true'])
		with open(os.path.join(dest, 'satellite_window_fill' + str(year) + '.pkl'), 'wb') as file:
			pkl.dump(lists, file, -1)

if __name__ == '__main__':
	start = time.time()
	source = '/home/kurora/satellite_data'
	dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satellite_window')
	os.makedirs(dest, exist_ok=True)
	make_window(source, dest, 1995, 2005, stop=48, step=6)
	make_window(source, dest, 2006, 2017, stop=48, step=6)
	print('Time: {0}'.format(time.time() - start), file=sys.stdout)
