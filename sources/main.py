import os
import pathlib

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import defaultdict

import load_data
import models
import elegans_som as es
import elegans_hmm as eh

from transformer import TransformerEncoder

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

data_gen = {
	0: load_data.som_timeseries_dataset,
	1: load_data.profile_timeseries_dataset
}


def hmm_experiments(pollutants, tinter_list, epochs)->'list':
	# HMM experiments
	hmm_accuracy = []
	for du in tinter_list:
		print(f'Observation interval: {du}')
		train_x, train_y, test_x, test_y = load_data.timeseries_for_hmm(pollutants, duration=du)
		models_dict = eh.model_training(train_x, n_iter=epochs, save=True)
		predict_dict = eh.predict(models_dict, test_x)
		accuracy, support = eh.metrics(0, predict_dict)
		hmm_accuracy.append(accuracy)
		print(f'HMM Accuracy {du}: {accuracy}\n\n')

	return hmm_accuracy


def dnn_experiments(dnns, pollutants, tinter_list, epochs)-> 'list of list':
	accuracy = []
	for model_id in dnns:
		model_accuracy = []
		for du in tinter_list:
			tf.keras.backend.clear_session()
			if tf.test.is_gpu_available('gpu'):
				print('GPU is available')
			data, model = models.do_experiment(
				data_gen[0],
				dnn_models[model_id],
				duration=du,
				epochs=epochs,
				data_set=pollutants,
				scaling=True
			)
			model_accuracy.append(model[2][1])
			print(f'{dnn_models[model_id].__name__} Accuracy {du}: {model[2][1]}\n\n')
		accuracy.append(model_accuracy)

	return accuracy


if __name__ == '__main__':
	tinter_list = [30, 60, 90, 120, 150, 180]  # observation interval(secs)s to predict the water condition
	pollutants = ['Benzen_0_1_ppm', 'Formaldehyde_0_1_ppm']
	epochs = 30
	dnns = [3] # list of dnn_models
	accuracy = defaultdict(list)
	num_experiments = 1 # 총 실험 횟수 -> 전체 실험의 평균 계산을 위해

	for _ in range(num_experiments):
		if True: # switch to include running hmm experiments
			print(f"{'='*10} HMM training start! {'='*10}")
			hmm_accuracy = hmm_experiments(pollutants, tinter_list, epochs=epochs)
			accuracy['hmm'].append(np.array(hmm_accuracy))
		print(f"{'='*10} DNN training start! {'='*10}")
		dnn_accuracy = dnn_experiments(dnns, pollutants, tinter_list, epochs=epochs)
		for i, acc in enumerate(dnn_accuracy):
			accuracy[dnn_models[dnns[i]].__name__].append(np.array(acc))

	accuracy = np.array(accuracy)
	avg_accuracy = np.mean(accuracy, axis=0)

	df = pd.DataFrame(avg_accuracy, index=tinter_list)
	fname = '_'.join([short_cut[p] for p in pollutants])
	fname += '_hmm_'
	fname += '_'.join([dnn_models[d].__name__ for d in dnns])
	fname += '.csv'
	fname = os.path.join('../result', fname)
	# if os.path.exists(fname):
	# 	os.remove(fname)
	df.to_csv(os.path.join('../results', fname), mode='a')