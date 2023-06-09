# Additional main to measure recall, precision and f1 etc.

import os
import pathlib
import time
import socket

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import defaultdict

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report

import load_data
import models
import elegans_som as es
import elegans_hmm as eh

from transformer import TransformerEncoder

import logging.config

logging.config.fileConfig('./logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

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


def hmm_predicts(pols: list, duration: int = 30)-> 'tuple of np.array':
	"""
	duration 동안의 7가지 패턴으로 구성된 테스트 데이터를 생성하고 학습된 HMM 모델을 불러 와서 예측

	:return:
	"""
	train_x, train_y, test_x, test_y = load_data.timeseries_for_hmm(pols, duration=duration)
	models_dict = eh.model_training(train_x, duration, n_components=5, n_iter=10, save=False)
	predict_dict = eh.predict(models_dict, test_x)

	y_true = np.array([])
	y_pred = np.array([])
	for key in predict_dict.keys(): # y_pred 와 y_true 사이 key의 순서 일치 반드시 확인
		y_true = np.concatenate((y_true, test_y[key]))
		y_pred = np.concatenate((y_pred, predict_dict[key]))

	return y_true, y_pred


def dnn_predicts_SOM_pattern(model_name: 'function',
							 pols: list,
							 data_gen: object,
							 duration: int = 30)-> 'tuple of np.array':
	"""
	duration 동안의 7가지 패턴으로 구성된 테스트 데이터를 생성하고
	학습된 deep learning 모델을 불러 와서 예측

	:param model: str
		h5 format 형태의 저장된 모델 path
	:param pols:
	:param duration:
	:return:
	"""
	train_x, train_y, valid_x, valid_y, test_x, test_y = load_data.som_timeseries_dataset(pols, duration=duration)
	d_names = [short_cut[name] for name in pols]
	scale = 'scale_1' #if scaling else 'scale_0'
	epochs = 30
	fname = '_'.join(d_names) + f'_du_{duration}_ep_{epochs}_{model_name.__name__}_{data_gen.__name__.split("_")[0]}_{scale}.h5'
	fbodies = fname.split('_')
	nclasses = len(pols) # the number of classes

	tf.keras.backend.clear_session()
	if tf.test.is_gpu_available('gpu'):
		print('GPU is available')

	#fname = 'test.h5'
	if 'transformer' in fbodies:
		# print('Model: Transformer ')
		try:
			model = tf.keras.models.load_model(os.path.join('../models', fname),
							custom_objects={"TransformerEncoder": TransformerEncoder})
		except ImportError as err:
			print('Model loading failure', err)
		except IOError as err:
			print("Model loading failure", err)
	else:
		try:
			model = model_name(train_x.shape[1], train_x.shape[-1], ncategory=nclasses)
			history = models.model_train(model, train_x, train_y, valid_x, valid_y, fname, epochs=30, ncategory=nclasses)
			model = tf.keras.models.load_model(os.path.join('../models', fname))
			print(f'{fname} is loaded.')
		except ImportError as err:
			print('Model loading failure', err)
		except IOError as err:
			print("Model loading failure", err)

	print('Model is being evaluated.\n\n')

	y_true = test_y
	print(model.evaluate(test_x, test_y))
	if nclasses == 2:
		y_pred = (model.predict(test_x) > 0.5).flatten()
	else:
		y_pred = np.argmax(model.predict(test_x), axis=1)
	#y_pred = np.array([0 if y < 0.5 else 1 for y in y_pred_2dim.flatten()])

	return y_true, y_pred, model

def metrics(y_true: np.array, y_pred: np.array):
	accuracy = accuracy_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred, average=None)
	precision = precision_score(y_true, y_pred, average=None)
	f1 = f1_score(y_true, y_pred, average=None)

	return accuracy, recall, precision, f1


if __name__ == '__main__':
	observations = [30, 60, 90, 120, 150, 180]  # observation interval(secs)s to predict the water condition
	pollutants = ['Normal', 'Formaldehyde_0_1_ppm'] #, 'Benzen_0_1_ppm']
	# epochs = 30
	# dnns = [3] # list of dnn_models
	# accuracy = defaultdict(list)
	num_experiments = 3 # 총 실험 횟수 -> 전체 실험의 평균 계산을 위해
	# hmm = True

	final_acc = list()
	final_re = list()
	final_pre = list()
	final_f1 = list()
	for _ in range(num_experiments):
		temp_acc = list()
		temp_re = list()
		temp_pre = list()
		temp_f1 = list()
		for du in observations:
			y_true, y_pred = hmm_predicts(pollutants, duration=du)
			accuracy, recall, precision, f1 = metrics(y_true, y_pred)
			temp_acc.append(accuracy)
			temp_re.append(recall)
			temp_pre.append(precision)
			temp_f1.append(f1)
		final_acc.append(temp_acc)
		final_re.append(temp_re)
		final_pre.append(temp_pre)
		final_f1.append(temp_f1)

	print(final_acc)
	print(final_re)
	print(final_pre)
	print(final_f1)

	### Very important facts ###
	# SOM 분석을 할 때마다 데이터에서 추출한 패턴이 새로 생성되므로
	# (예를 들면 첫 번째 데이터 생성에서는 flat 패턴이 1번이라면 다음 번 생성에서는 3번으로 변경됨)
	# 특정 데이터에 학습된 모델을 불러와 새로 SOM 분석을 통해 생성한 데이터를 대상으로 사용하면
	# 엉뚱한 결과과 나옮
	# 따라서, 한 번 SOM 분석을 하면 클러스터링 후 클러스터의 중심을 저장해 두고 신규 데이터와의 거리를 계산해
	# 패턴 라벨링을 하는 방법을 사용하여야 할 것으로 생각됨 -> 구현 필요
	# 당장은 SOM 분석 후 데이터를 저장해 두고 이를 재사용하는 방법을 사용하여야 함

	# y_true, y_pred, model = dnn_predicts_SOM_pattern(dnn_models[3], pollutants, data_gen[0], duration=30)
	# acc, rec, pre, f1 = metrics(y_true, y_pred)
	# print(f'{acc}, {rec}, {pre}, {f1}')