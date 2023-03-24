import os
import pathlib
import time
import socket

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import defaultdict

import load_data
import models
import elegans_som as es
import elegans_hmm as eh

from transformer import TransformerEncoder

import logging.config

logging.config.fileConfig('./logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# dfiles = {
# 	'Form_01': 'data/Formaldehyde_0_1_ppm',
# 	'Form_05': 'data/Formaldehyde_0_1_ppm',
# 	'Normal': 'data/Normal',
# 	'Benzen_01': 'data/Benzen_0_1_ppm',
# 	'Benzen_05': 'data/Benzen_0_5_ppm',
# 	'Toulen_01': 'data/Toulen_0_1_ppm',
# 	'Toulen_05': 'data/Toulen_0_5_ppm'
# }

short_cut = {
	'Formaldehyde_0_1_ppm': 'Form_01',
	'Formaldehyde_0_5_ppm': 'Form_05',
	'Normal'              : 'Normal',
	'Benzen_0_1_ppm'      : 'Benzen_01',
	'Benzen_0_5_ppm'      : 'Benzen_05',
	'Toluen_0_1_ppm'      : 'Toluen_01',
	'Toluen_0_5_ppm'      : 'Toluen_05'
}

target = {
	'Normal'              : 0,
	'Benzen_0_1_ppm'      : 1,
	'Benzen_0_5_ppm'      : 2,
	'Formaldehyde_0_1_ppm': 3,
	'Formaldehyde_0_5_ppm': 4,
	'Toluen_0_1_ppm'      : 5,
	'Toluen_0_5_ppm'      : 6
}

dnn_models = {
	0: models.simple_lstm,
	1: models.densely_connected,
	2: models.conv_1d,
	3: models.lstm,
	4: models.gru,
	5: models.bidirectional_lstm,
	6: models.transformer
}

# Selection of data type: SOM pattern or Bls profile
data_gen = {
	0: load_data.som_timeseries_dataset,
	1: load_data.profile_timeseries_dataset
}


def hmm_experiments(pollutants, tinter_list, epochs)->'list':
	# HMM experiments
	hmm_accuracy = []
	for du in tinter_list:
		logger.info(f'Observation interval: {du}')
		print(f'Observation interval: {du}')
		train_x, train_y, test_x, test_y = load_data.timeseries_for_hmm(pollutants, duration=du)
		models_dict = eh.model_training(train_x, du, n_iter=epochs, save=True)
		predict_dict = eh.predict(models_dict, test_x)
		accuracy, support = eh.metrics(0, predict_dict)
		hmm_accuracy.append(accuracy)
		logger.info(f'HMM Accuracy {du}: {accuracy}\n\n')

	return hmm_accuracy


def dnn_experiments(dnns, data_type, pollutants, tinter_list, epochs, train=True, scaling=True)-> 'list of list':
	"""

	:param dnns:
	:param data_type:
		SOM data type: 0 과 BLS profile data type: 1 중 선택
	:param pollutants:
	:param tinter_list:
	:param epochs:
	:return: accuracy
		각 모델들을 대상으로 tinter_list 즉 duration에 대한 정확도 반환
		[[30, 60, 90, 120, 150, 180], [30, 60, 90, 120, 150, 180], [30, 60, 90, 120, 150, 180] ...]
	"""
	accuracy = []
	for model_id in dnns:
		model_accuracy = [] # for each duration
		for du in tinter_list:
			logger.info(f'Observation interval: {du}')
			print(f'Observation interval: {du}')
			tf.keras.backend.clear_session()
			if tf.test.is_gpu_available('gpu'):
				print('GPU is available')
			data, model = models.do_experiment(
				data_gen[data_type], # data type <- SOM pattern: 0, BLS profile: 1
				dnn_models[model_id],
				duration=du,
				epochs=epochs,
				data_set=pollutants,
				train=train,
				scaling=scaling
			)
			model_accuracy.append(model[2][1]) #
			logger.info(f'{dnn_models[model_id].__name__} Accuracy {du}: {model[2][1]}\n\n')
		accuracy.append(model_accuracy)

	return accuracy


def display(data):
	# hmm = np.array([0.53310349, 0.57035034, 0.56001052, 0.59778913, 0.5781693, 0.57265539])
	# lstm = np.array([0.84932139, 0.90554857, 0.93012849, 0.94313756, 0.95631149, 0.9589441])
	hmm = data.iloc[0]
	lstm = data.iloc[1]

	duration = [30, 60, 90, 120, 150, 180]
	ticks = range(len(duration))

	plt.rcParams['font.size'] = 17
	plt.rcParams['font.weight'] = 'bold'
	params = {'linewidth': 2.0, 'markersize': 12}

	fig, ax = plt.subplots(1, 1, figsize=(10, 7))
	ax.plot(
		ticks, hmm, 'bd--',
		linewidth=params['linewidth'], markersize=params['markersize'],
		label='Hidden Markov Model'
	)
	ax.plot(
		ticks, lstm, 'ko-',
		linewidth=params['linewidth'], markersize=params['markersize'],
		label='Long Short-Term Memory'
	)
	ax.set_xlabel('Duration', fontsize=20, fontdict=dict(weight='bold'))
	ax.set_ylabel('Accuracy', fontsize=20, fontdict=dict(weight='bold'))
	ax.set_xticks(ticks, duration)
	ax.set_title("Accuracy Comparison", fontsize=24, fontdict=dict(weight='bold'))
	ax.legend()


if __name__ == '__main__':
	tinter_list = [180] #[30, 60, 90, 120, 150] #, 180]  # observation interval(secs)s to predict the water condition
	pollutants = ['Normal', 'Formaldehyde_0_1_ppm']
	epochs = 30
	dnns = [3] # list of dnn_models
	accuracy = defaultdict(list)
	num_experiments = 3 # 총 실험 횟수 -> 전체 실험의 평균 계산을 위해
	hmm = True

	for _ in range(num_experiments):
		if hmm: # switch to include running hmm experiments
			print(f"{'='*10} HMM training start! {'='*10}")
			hmm_accuracy = hmm_experiments(pollutants, tinter_list, epochs=epochs)
			accuracy['hmm'].append(np.array(hmm_accuracy))
		# print(f"{'='*10} DNN training start! {'='*10}")
		# dnn_accuracy = dnn_experiments(dnns, 0, pollutants, tinter_list, epochs=epochs)
		# for i, acc in enumerate(dnn_accuracy):
		# 	accuracy[dnn_models[dnns[i]].__name__].append(np.array(acc))

	avg_accuracy = []
	for m in accuracy.keys():
		avg_accuracy.append(np.mean(np.asarray(accuracy[m]), axis=0))

	index = []
	if hmm:
		index.append('hmm')
	index += [dnn_models[d].__name__ for d in dnns]
	df = pd.DataFrame(avg_accuracy, columns=tinter_list, index=index)
	fname = '_'.join([short_cut[p] for p in pollutants])
	fname += '_'
	if hmm:
		fname += 'hmm_'
	fname += '_'.join([dnn_models[d].__name__ for d in dnns])
	fname += '_accuracy_' + time.strftime('%Y%m%d%H%M', time.localtime()) \
			 + socket.gethostbyname(socket.gethostname()).split('.')[-1] + '.csv'
	# fname = os.path.join('../result', fname)
	# if os.path.exists(fname):
	# 	os.remove(fname)
	df.to_csv(os.path.join('../results', fname), mode='a')
