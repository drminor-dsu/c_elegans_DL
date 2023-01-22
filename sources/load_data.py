import os
import sys
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import json

from sklearn.utils import shuffle
from tensorflow import keras

import elegans_som as es
import elegans_hmm as eh

sys.path.extend(['./'])  # for module import elegans_som, _hmm: 복사해온 경우 파일 import가 안되는 경우

# directory = {'Form_01': 'Formaldehyde_0_1_ppm',
# 			'Form_05': 'Formaldehyde_0_5_ppm',
# 			'Normal': 'Normal',
# 			'Benzen_01': 'Benzen_0_1_ppm',
# 			'Benzen_05': 'Benzen_0_5_ppm',
# 			'Toluen_01': 'Toluen_0_1_ppm',
# 			'Toluen_05': 'Toluen_0_5_ppm'}

target = {
	'Normal': 0,
	'Benzen_0_1_ppm': 1,
	'Benzen_0_5_ppm': 2,
	'Formaldehyde_0_1_ppm': 3,
	'Formaldehyde_0_5_ppm': 4,
	'Toluen_0_1_ppm': 5,
	'Toluen_0_5_ppm': 6
}

data_path = os.path.join(pathlib.Path(__file__).parents[1], 'data')
# '../../data/'
train_rate = 0.6  # the ratio of training data
valid_rate = 0.2  # the ratio of validation data
test_rate = 0.2  # the ratio of test data (== remaining data except train data + valid data


def timeseries_for_hmm(
		pollutants: list,
		start=10,
		end=40,
		duration=30
	) -> 'train_x, train_y, valid_x, valid_y, test_x, test_y: dictionary':
	'''
	Hidden Markov Model을 위한 데이터 생성
	HMM은 오염물질 각각에 대해 별개의 모델을 만들어 학습해야 하므로 딕셔너리를 활용해 각 화학물질별 데이터를 분리해서 생

	:param pollutants:
	:param start:
	:param end:
	:param duration:
	:return: train_x, train_y, valid_x, valid_y, test_x, test_y: dictionary

	'''

	# data = es.load_all_data(*kinds, start=start, end=end)  # 단위(분) ex) start 10분, end 40분: 30분 데이터 사용
	raw_data, num_files = es.load_data(pollutants, start=start, end=end)

	# SOM을 이용한 패턴 종류 결정 및 데이터 패턴 시퀀스로 변경
	seqs = es.som_analysis(raw_data, n_columns=14, n_rows=14, n_cluster=7)

	# 화학물질 별 선택된 개체들의 데이터 분리
	divided_seq = es.divide_sequences(seqs, pollutants, num_files, start=start, end=end)

	data = defaultdict(list)  # data points:
	classes = defaultdict(list)  # target
	num_frames = dict()  # the number of frames for each pollutant
	num_samples = dict()  # the number of data_sets points with the size of sequence length for each pollutant
	start_frame = start * 60 * 4  # start(min) x 60(sec) x 4(frame per second)
	end_frame = end * 60 * 4
	sequence_length = duration * 4

	print(pollutants)
	target = dict()
	for i in range(len(pollutants)):
		target[pollutants[i]] = i

	num_samples_per_file = 60 * 4 * (end - start)  # 한 파일에 포함된 sequence 수
	pol_id = 0
	for pol in divided_seq:
		n_frame = 0
		n_sample = 0
		for i in range(num_files[pol]):
			fseqs = divided_seq[pol][i * num_samples_per_file: (i + 1) * num_samples_per_file]
			targets = np.full(len(fseqs), target[pol])
			n_frame += len(fseqs)
			# print(pol, targets[0], fseqs.shape, targets.shape)
			time_series = keras.utils.timeseries_dataset_from_array(
				fseqs,
				targets=targets,
				sequence_length=sequence_length,
				shuffle=False,
				batch_size=None
			)
			for x, y in time_series:
				data[pol].append(x.numpy())
				classes[pol].append(y.numpy())
				n_sample += 1

		num_frames[pol] = n_frame
		num_samples[pol] = n_sample
		print(f'{pollutants[pol_id]} -> Number of frames: {n_frame}, Number of samples: {n_sample}')
		pol_id += 1

	train_x = dict();
	train_y = dict()
	test_x = dict();
	test_y = dict()

	for key in data.keys():
		n_train = int(len(data[key]) * (train_rate + valid_rate))  # the number of training data samples
		x_data, y_data = shuffle(np.array(data[key]), np.array(classes[key]))
		train_x[key] = x_data[:n_train]
		train_y[key] = y_data[:n_train]
		test_x[key] = x_data[n_train:]
		test_y[key] = y_data[n_train:]
		print(f"{key} -> train: {n_train}, test: {len(data[key]) - n_train} : no validation for HMM")

	return train_x, train_y, test_x, test_y


def som_timeseries_dataset(
		pollutants: list,
		start=10,
		end=40,
		duration=30,
		scaling=True
	) -> 'train_x, train_y, valid_x, valid_y, test_x, test_y':
	"""

	:param pollutants:
	:param start:
	:param end:
	:param duration:
	:param scaling:
		실제 의미 없음. 데이터 로딩 함수들의 인터페이스를 동일하게 하기 위해 추가해 놓은 인수임.
	:return:
	"""

	# data = es.load_all_data(*kinds, start=start, end=end)  # 단위(분) ex) start 10분, end 40분: 30분 데이터 사용
	raw_data, num_files = es.load_data(pollutants, start=start, end=end)

	# SOM을 이용한 패턴 종류 결정 및 데이터 패턴 시퀀스로 변경
	seqs = es.som_analysis(raw_data, n_columns=14, n_rows=14, n_cluster=7)

	# 화학물질 별 선택된 개체들의 데이터 분리
	divided_seq = es.divide_sequences(seqs, pollutants, num_files, start=start, end=end)

	data = defaultdict(list)  # data points:
	classes = defaultdict(list)  # target
	num_frames = dict()  # the number of frames for each pollutant
	num_samples = dict()  # the number of data_sets points with the size of sequence length for each pollutant
	start_frame = start * 60 * 4  # start(min) x 60(sec) x 4(frame per second)
	end_frame = end * 60 * 4
	sequence_length = duration * 4

	print(pollutants)
	target = dict()
	for i in range(len(pollutants)):
		target[pollutants[i]] = i

	num_samples_per_file = 60 * 4 * (end - start)  # 한 파일에 포함된 sequence 수
	pol_id = 0
	for pol in divided_seq:
		n_frame = 0
		n_sample = 0
		for i in range(num_files[pol]):
			fseqs = divided_seq[pol][i * num_samples_per_file: (i + 1) * num_samples_per_file]
			targets = np.full(len(fseqs), target[pol])
			n_frame += len(fseqs)
			# print(pol, targets[0], fseqs.shape, targets.shape)
			time_series = keras.utils.timeseries_dataset_from_array(
				fseqs,
				targets=targets,
				sequence_length=sequence_length,
				shuffle=False,
				batch_size=None
			)
			for x, y in time_series:
				data[pol].append(x.numpy())
				classes[pol].append(y.numpy())
				n_sample += 1

		num_frames[pol] = n_frame
		num_samples[pol] = n_sample
		print(f'{pollutants[pol_id]} -> Number of frames: {n_frame}, Number of samples: {n_sample}')
		pol_id += 1

	train_x = [];
	train_y = []
	valid_x = [];
	valid_y = []
	test_x = [];
	test_y = []

	for key in data.keys():
		n_train = int(len(data[key]) * train_rate)  # the number of training data samples
		n_valid = int(len(data[key]) * valid_rate)  # the number of validation data samples
		x_data, y_data = shuffle(np.array(data[key]), np.array(classes[key]))
		train_x.append(x_data[:n_train])
		train_y.append(y_data[:n_train])
		valid_x.append(x_data[n_train:n_train + n_valid])
		valid_y.append(y_data[n_train:n_train + n_valid])
		test_x.append(x_data[n_train + n_valid:])
		test_y.append(y_data[n_train + n_valid:])

	train_x = np.concatenate(train_x, axis=0)
	train_y = np.concatenate(train_y, axis=0)
	valid_x = np.concatenate(valid_x, axis=0)
	valid_y = np.concatenate(valid_y, axis=0)
	test_x = np.concatenate(test_x, axis=0)
	test_y = np.concatenate(test_y, axis=0)

	train_x, train_y = shuffle(train_x, train_y)
	valid_x, valid_y = shuffle(valid_x, valid_y)
	test_x, test_y = shuffle(test_x, test_y)

	train_x = np.expand_dims(train_x, axis=2).astype(np.float32)
	valid_x = np.expand_dims(valid_x, axis=2).astype(np.float32)
	test_x = np.expand_dims(test_x, axis=2).astype(np.float32)
	print(f"train: {train_y.shape}, valid: {valid_y.shape}, test: {test_y.shape}")

	return train_x, train_y, valid_x, valid_y, test_x, test_y


def profile_timeseries_dataset(
		pollutants: list,
		start=10,
		end=40,
		duration=30,
		scaling=True
	) -> 'train_x, train_y, valid_x, valid_y, test_x, test_y':
	"""
	오염 물질들의 데이터 파일 전체를 keras.utils.timeseries_dataset_from_array를 이용해 time series dataset 으로 변환
	train_x, train_y, valid_x, valid_y, test_x, test_y 반

	:param pollutants: list
		ex) ['Formaldehyde_0_1_ppm', 'Normal']
	:param start: int
		The default is 10(분). 60분 짜리 csv 파일에서 실제 데이터로 사용할 부분의 시작 시각
	:param end: int
		The default is 40(분). 60분 짜리 csv 파일에서 실제 데이터로 사용할 부분의 마지막 시각
	:param duration: int
		The default is 30(초). 학습 및 판별에 사용할 관찰 시간
	:parma scaling: Bool
		The default is True.

	:return: tuple of splitted data set
		train_x, train_y, valid_x, valid_y, test_x, test_y
	"""

	data = defaultdict(list)  # data points:
	classes = defaultdict(list)  # target
	num_frames = dict()  # the number of frames for each pollutant
	num_samples = dict()  # the number of data_sets points with the size of sequence length for each pollutant
	start_frame = start * 60 * 4  # start(min) x 60(sec) x 4(frame per second)
	end_frame = end * 60 * 4
	sequence_length = duration * 4

	print(pollutants)
	target = dict()
	for i in range(len(pollutants)):
		target[pollutants[i]] = i

	for pollutant in pollutants:
		directory = os.path.join(data_path, pollutant)
		files = os.listdir(directory)
		files.sort()
		n_frame = 0
		n_sample = 0
		for csvfile in files:
			print(csvfile)
			df = pd.read_csv(os.path.join(directory, csvfile), header=None)
			df.dropna(axis=0, inplace=True)
			df = df[start_frame: end_frame]
			n_frame += df.shape[0]

			# keras.utils.timeseries_dataset_from_array 사용 부분 추후 효율적인 코드로 고칠 것 ! shuffle True 반영
			targets = np.full(len(df), target[pollutant])
			timeseries = keras.utils.timeseries_dataset_from_array(
				df.values,
				targets=targets,
				sequence_length=sequence_length,
				shuffle=False,
				batch_size=None
			)
			for x, y in timeseries:
				data[pollutant].append(x.numpy())
				classes[pollutant].append(y.numpy())
				n_sample += 1

		num_frames[pollutant] = n_frame
		num_samples[pollutant] = n_sample
		print(f'{pollutant} -> Number of frames: {n_frame}, Number of samples: {n_sample}')
		i += 1

	train_x = [];
	train_y = []
	valid_x = [];
	valid_y = []
	test_x = [];
	test_y = []

	for key in data.keys():
		n_train = int(len(data[key]) * train_rate)  # the number of training data samples
		n_valid = int(len(data[key]) * valid_rate)  # the number of validation data samples
		x_data, y_data = shuffle(np.array(data[key]), np.array(classes[key]))
		train_x.append(x_data[:n_train])
		train_y.append(y_data[:n_train])
		valid_x.append(x_data[n_train:n_train + n_valid])
		valid_y.append(y_data[n_train:n_train + n_valid])
		test_x.append(x_data[n_train + n_valid:])
		test_y.append(y_data[n_train + n_valid:])

	train_x = np.concatenate(train_x, axis=0)
	train_y = np.concatenate(train_y, axis=0)
	valid_x = np.concatenate(valid_x, axis=0)
	valid_y = np.concatenate(valid_y, axis=0)
	test_x = np.concatenate(test_x, axis=0)
	test_y = np.concatenate(test_y, axis=0)

	train_x, train_y = shuffle(train_x, train_y)
	valid_x, valid_y = shuffle(valid_x, valid_y)
	test_x, test_y = shuffle(test_x, test_y)

	if scaling:
		train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
		valid_x = (valid_x - valid_x.mean(axis=0)) / valid_x.std(axis=0)
		test_x = (test_x - test_x.mean(axis=0)) / test_x.std(axis=0)
	print(f"train: {train_y.shape}, valid: {valid_y.shape}, test: {test_y.shape}")

	return train_x, train_y, valid_x, valid_y, test_x, test_y


# if __name__ == '__main__':
# 	train_x, train_y, valid_x, valid_y, test_x, test_y = som_timeseries_dataset(['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm'])
# train_x, train_y, valid_x, valid_y, test_x, test_y = profile_timeseries_dataset(['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm'])
