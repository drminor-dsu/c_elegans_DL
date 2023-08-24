# Additional main to measure recall, precision and f1 etc.

import os
import pathlib
import time
import socket
import csv
import re
import json

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


def hmm_predicts(pols: list, duration: int = 30)->'tuple of np.array':
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


def dnn_predicts(model_name: object,
                             pols:list,
                             data_type:object=0,
                             duration:int=30,
                             epochs:int=30)-> 'tuple of np.array':
    """
    duration 동안의 7가지 패턴으로 구성된 테스트 데이터를 생성하고
    학습된 deep learning 모델을 불러 와서 예측

    :param model: str
        h5 format 형태의 저장된 모델 path
    :param pols: list
    :param duration: defalut value 30
        관찰 시간
    :param epochs: 학습 횟수
    :return: tuple of np.array
    """
    if data_type == 0:
        train_x, train_y, valid_x, valid_y, test_x, test_y = data_gen[data_type](pols, duration=duration)
    else:
        train_x, train_y, valid_x, valid_y, test_x, test_y = data_gen[data_type](pols, duration=duration)
    d_names = [short_cut[name] for name in pols]
    scale = 'scale_1' #if scaling else 'scale_0'
    fname = '_'.join(d_names) + f'_du_{duration}_ep_{epochs}_{model_name.__name__}_{data_gen[data_type].__name__.split("_")[0]}_{scale}.h5'
    fbodies = fname.split('_')
    nclasses = len(pols) # the number of classes

    tf.keras.backend.clear_session()
    if tf.test.is_gpu_available('gpu'):
        print('GPU is available')

    if 'transformer' in fbodies:
        # print('Model: Transformer ')
        try:
            model = model_name(train_x.shape[1], train_x.shape[-1], ncategory=nclasses)
            history = models.model_train(model, train_x, train_y, valid_x, valid_y, fname, epochs=30,
                                         ncategory=nclasses)
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

    print(f'{model_name.__name__} is being evaluated.\n\n')

    y_true = test_y
    #print(model.evaluate(test_x, test_y))
    if nclasses == 2:
        y_pred = (model.predict(test_x) > 0.5).flatten()
    else:
        y_pred = np.argmax(model.predict(test_x), axis=1)

    return y_true, y_pred


def metrics(y_true: np.array, y_pred: np.array):
    """
    Accuracy, Recall, Precision, and F1 score are returned.
    :param y_true : np.array
    :param y_pred : np.array
    :return:
    accuracy : list
    recall : list
    precision : list
    f1 : list
    """
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    return accuracy, recall, precision, f1


def file_name(pollutants, d=3):
    fname = '_'.join([short_cut[p] for p in pollutants])
    fname += '_' + dnn_models[d].__name__ + '_hidden_3_'
    fname += '_' + time.strftime('%Y%m%d%H%M', time.localtime()) \
             + socket.gethostbyname(socket.gethostname()).split('.')[-1] + '.csv'

    return fname

def new_lstm_experiments():
    model_name = models.lstm
    observations = [30, 60, 90, 120, 150, 180]
    metric = ['accuracy', 'recall', 'precision', 'f1_score']
    pol_list = [['Normal', 'Formaldehyde_0_1_ppm'], ['Normal', 'Benzen_0_1_ppm'],
                ['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm'], ['Normal', 'Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm']]
    for pol in pol_list:
        scores = do_job_dnn(pol, observations)
        lstm_metrics = dict()
        df = pd.DataFrame()
        for m in metric:
            lstm_metrics[m] = scores[m].mean(axis=0)
            if m != 'accuracy':
                for i in range(lstm_metrics[m].shape[-1]):
                    df['lstm_' + m + '_' + str(i)] = lstm_metrics[m][:, i]
            else:
                df['lstm_' + m] = lstm_metrics[m]
        df.index = observations
        fname = file_name(pol)

        try:
            df.to_csv(pathlib.Path(__file__).parents[1].joinpath(f'results/{fname}'))
        except:
            print("Cannot save file")
            return df
    return df


def do_job_dnn(pollutants:list, observations:list, num: int = 3) -> np.array:
    ### Very important facts ###
    # SOM 분석을 할 때마다 데이터에서 추출한 패턴이 새로 생성되므로
    # (예를 들면 첫 번째 데이터 생성에서는 flat 패턴이 1번이라면 다음 번 생성에서는 3번으로 변경됨)
    # 특정 데이터에 학습된 모델을 불러와 새로 SOM 분석을 통해 생성한 데이터를 대상으로 사용하면
    # 엉뚱한 결과과 나옮
    # 따라서, 한 번 SOM 분석을 하면 클러스터링 후 클러스터의 중심을 저장해 두고 신규 데이터와의 거리를 계산해
    # 패턴 라벨링을 하는 방법을 사용하여야 할 것으로 생각됨 -> 구현 필요
    # 당장은 SOM 분석 후 데이터를 저장해 두고 이를 재사용하는 방법을 사용하여야 함

    final_acc = list()
    final_re = list()
    final_pre = list()
    final_f1 = list()
    for n in range(num):
        temp_acc = list()
        temp_re = list()
        temp_pre = list()
        temp_f1 = list()
        for du in observations:
            print(f'#{n + 1}: prediction starts for duration {du}')
            y_true, y_pred = dnn_predicts(dnn_models[3], pollutants, 0, duration=du)
            accuracy, recall, precision, f1 = metrics(y_true, y_pred)
            temp_acc.append(accuracy)
            temp_re.append(recall)
            temp_pre.append(precision)
            temp_f1.append(f1)
        final_acc.append(temp_acc)
        final_re.append(temp_re)
        final_pre.append(temp_pre)
        final_f1.append(temp_f1)

    scores = dict()

    scores.update(
        {'accuracy': np.asarray(final_acc),
        'recall': np.asarray(final_re),
        'precision': np.asarray(final_pre),
        'f1_score': np.asarray(final_f1)}
    )

    return scores

    # with open('./metrics_lstm.csv', 'w') as fp:
    #     writer = csv.DictWriter(fp, fieldnames=scores.keys())
    #     writer.writeheader()
    #     writer.writerow(scores)

def do_job_HMM(pollutants:list, observations:list, num:int=3)->'tuple of ndarray':
    # epochs = 30
    # dnns = [3] # list of dnn_models
    # accuracy = defaultdict(list)
    num_experiments = num
    # hmm = True

    final_acc = list()
    final_re = list()
    final_pre = list()
    final_f1 = list()
    for n in range(num_experiments):
        temp_acc = list()
        temp_re = list()
        temp_pre = list()
        temp_f1 = list()
        for du in observations:
            print(f'#{n + 1}: prediction starts for duration {du}')
            y_true, y_pred = hmm_predicts(pollutants, duration=du)
            accuracy, recall, precision, f1 = metrics(y_true, y_pred)
            print(f'{accuracy}, {recall}, {precision}, {f1}')
            temp_acc.append(accuracy)
            temp_re.append(recall)
            temp_pre.append(precision)
            temp_f1.append(f1)
        final_acc.append(temp_acc)
        final_re.append(temp_re)
        final_pre.append(temp_pre)
        final_f1.append(temp_f1)

    return np.asarray(final_acc), np.asarray(final_re), np.asarray(final_pre), np.asarray(final_f1)


def metrics_mean(scores: list, metric: str) -> np.array:
    """
    csv.DictWriter로 ndarray를 value로 갖는 dict를 저장하였더니, 문자열로 저장되어서,
    이를 파싱하고 평균을 계산하여 반환
    scores['metric'] = np.array()

    :param scores: csv 파일에서 읽어온 리스트
    :param metric: 평가 지표, 'accuracy', 'recall', 'precision', 'f1_score' 중 하나
    :return: metric 지표값을 ndarray로 변환하여 반환
    """

    score = json.loads(re.sub(' ', ',', re.sub(r'\s\s+', ' ', re.sub(', ', ',', re.sub('\n', '', re.sub('\n ', ',', re.sub('\s+]', ']', scores[0][metric])))))))

    return np.asarray(score).mean(axis=0)


def drawing(hmm, lstm, metric):
    """
    The purpose of this drawing function is for the test.
    For publishing edition, refer to graph.py located in Windows.

    :param hmm:
    :param lstm:
    :param metric:
    :return:
    """
    if metric != 'accuracy':
        hmm_mean = hmm[metric][:, 1]
        lstm_mean = lstm[metric][:, 1]
    else:
        hmm_mean = hmm[metric]
        lstm_mean = lstm[metric]

    duration = [30, 60, 90, 120, 150, 180]
    ticks = range(len(duration))

    plt.rcParams['font.size'] = 17
    plt.rcParams['font.weight'] = 'bold'
    params = {'linewidth': 2.0, 'markersize': 12}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(
        ticks, hmm_mean, 'bd--',
        linewidth=params['linewidth'], markersize=params['markersize'],
        label='Hidden Markov Model'
    )
    ax.plot(
        ticks, lstm_mean, 'ko-',
        linewidth=params['linewidth'], markersize=params['markersize'],
        label='Long Short-Term Memory'
    )
    ax.set_xlabel('Duration', fontsize=20, fontdict=dict(weight='bold'))
    ax.set_ylabel(metric.title(), fontsize=20, fontdict=dict(weight='bold'))
    ax.set_xticks(ticks, duration)
    ax.set_title(f"{metric.title()} Comparison", fontsize=24, fontdict=dict(weight='bold'))
    ax.legend()

def display_metrics(metric:list):
    """
    HMM 과 LSTM 사이의 주어진 metric 그림 저장
    :return:
    """

    with open("./metrics_HMM.csv", 'r') as fp:
        reader = csv.DictReader(fp)
        hmm_scores = list(reader)

    with open('./metrics_lstm.csv', 'r') as fp:
        reader = csv.DictReader(fp)
        lstm_scores = list(reader)

    hmm_metrics = dict()
    lstm_metrics = dict()
    df = pd.DataFrame()
    for m in metric:
        hmm_metrics[m] = metrics_mean(hmm_scores, m)
        lstm_metrics[m] = metrics_mean(lstm_scores, m)

        if m != 'accuracy':
            for i in range(hmm_metrics[m].shape[-1]):
                df['hmm_'+m+'_'+str(i)] = hmm_metrics[m][:, i]
                df['lstm_' + m + '_' + str(i)] = lstm_metrics[m][:, i]
        else:
            df['hmm_'+m] = hmm_metrics[m]
            df['lstm_' + m] = lstm_metrics[m]

        #drawing(hmm_metrics, lstm_metrics, m)

    df.index = [30, 60, 90, 120, 150, 180]

    # df.to_csv('./metrics_plots.csv')

    return df


def rnn_compare(metric, opservation) -> pd.DataFrame:
    """
    RNN 사이의 성능 비교 그래프를 그리기 위해 평균을 계산하고 DataFrame 형식으로 반환
    :param metric:
    :param opservation:
    :return:
    """
    fname = './dnn_compare_unified__hidden_plot.csv'
    if pathlib.Path(fname).exists():
        print("Please Check file name. The file with the same name already exists.")
        return

    with open("./metrics_lstm_hidden3.csv", "r") as fp1:
        reader = csv.DictReader(fp1)
        lstm_scores = list(reader)
    with open("./metrics_gru.csv", "r") as fp2:
        reader = csv.DictReader(fp2)
        gru_scores = list(reader)
    with open("./metrics_bidirectional_lstm.csv", "r") as fp3:
        reader = csv.DictReader(fp3)
        bi_lstm_scores = list(reader)
    # with open("./metrics_transformer.csv", "r") as fp:
    #     reader = csv.DictReader(fp)
    #     transformer_scores = list(reader)

    lstm_metrics = dict()
    gru_metrics = dict()
    bi_lstm_metrics = dict()
    #transformer_metrics = dict()

    df = pd.DataFrame()
    for m in metric:
        lstm_metrics[m] = metrics_mean(lstm_scores, m)
        gru_metrics[m] = metrics_mean(gru_scores, m)
        bi_lstm_metrics[m] = metrics_mean(bi_lstm_scores, m)
        #transformer_metrics[m] = metrics_mean(transformer_scores, m)

        if m != 'accuracy':
            for i in range(lstm_metrics[m].shape[-1]):
                df['LSTM_' + m + '_' + str(i)] = lstm_metrics[m][:, i]
                df['GRU_' + m + '_' + str(i)] = gru_metrics[m][:, i]
                df['Bi-LSTM_' + m + '_' + str(i)] = bi_lstm_metrics[m][:, i]
                #df['Transformer_' + m + '_' + str(i)] = transformer_metrics[m][:, i]
        else:
            df['LSTM_' + m] = lstm_metrics[m]
            df['GRU_' + m] = gru_metrics[m]
            df['Bi-LSTM_' + m] = bi_lstm_metrics[m]
            #df['Transformer_' + m] = transformer_metrics[m]

    # drawing(hmm_metrics, lstm_metrics, m)

    df.index = [30, 60, 90, 120, 150, 180]

    df.to_csv(fname)

    return df


if __name__ == '__main__':
    observations = [30, 60, 90, 120, 150, 180]  # observation interval(secs)s to predict the water condition
    pollutants = ['Normal', 'Formaldehyde_0_1_ppm']
    num_experiments = 3 # 총 실험 횟수 -> 전체 실험의 평균 계산을 위해
    metric = ['accuracy', 'recall', 'precision', 'f1_score']
    # epochs = 30
    # dnns = [3] # list of dnn_models

    accuracy, recall, precision, f1 = do_job_HMM(pollutants, observations, num_experiments)
    score = do_job_dnn(pollutants, observations, num_experiments)

    #do_job_HMM(pollutants, observations, num_experiments)
    #do_job_HMM(pollutants, observations, num_experiments)

    #df = display_metrics(metric=metric)

    # for rnn comparison with different "metrics_....csv" files
    # df = rnn_compare(metric, observations)
    # df = new_lstm_experiments()