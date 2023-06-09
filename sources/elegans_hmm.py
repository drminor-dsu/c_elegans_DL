import os
import sys
sys.path.extend('./')

import pathlib
import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from hmmlearn import hmm
import pickle
import time

import load_data

droot = pathlib.Path(__file__).parents[0]
dfiles = {
    'Form_01': 'data/Formaldehyde_0_1_ppm',
    'Form_05': 'data/Formaldehyde_0_1_ppm',
    'Normal': 'data/Normal',
    'Benzen_01': 'data/Benzen_0_1_ppm',
    'Benzen_05': 'data/Benzen_0_5_ppm',
    'Toulen_01': 'data/Toulen_0_1_ppm',
    'Toulen_05': 'data/Toulen_0_5_ppm'
}


def split_input_data_from_files(*kinds, test_size=0.2):
    print('norm data reading')
    data_dict = OrderedDict()
    for chemical in kinds:
        print('Loading {}'.format(chemical))
        seq = pd.read_csv(os.path.join(droot, dfiles[chemical]), dtype=np.int32, header=None)
        data_dict[chemical] = seq

    train_dict = OrderedDict()
    test_dict = OrderedDict()
    for d in data_dict:
        train, test = train_test_split(data_dict[d], test_size=test_size, random_state=42)
        train_dict[d] = train
        test_dict[d] = test

    return train_dict, test_dict


def split_input_data_from_dicts(data_dict, test_size=0.2):
    train_dict = OrderedDict()
    test_dict = OrderedDict()
    for d in data_dict:
        train, test = train_test_split(data_dict[d], test_size=test_size, random_state=42)
        train_dict[d] = train
        test_dict[d] = test
        # print(np.asarray(train).shape)
        # print(np.asarray(test).shape)

    return train_dict, test_dict


def build_and_save_hmm(x, label, du, n_components=5, n_iter=10, save=False):
    model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter)
    model.fit(x)

    if save:
        #ftime = time.strftime('%y%m%d%H%M', time.localtime())
        fname = f'../models/{label}_du_{du}_hmm.pkl'
        with open(fname, 'wb') as fd:
            pickle.dump(model, fd)

    return model


def model_training(train_dict, du, n_components=5, n_iter=10, save=False):

    model_dicts = OrderedDict()
    for chemical in train_dict:
        print('Training {} HMM'.format(chemical))
        hmm = build_and_save_hmm(train_dict[chemical], chemical, du, n_components=n_components, n_iter=n_iter, save=save)
        model_dicts[chemical] = hmm

    return model_dicts


def load_hmm(label, du):
    fname = f'../models/{label}_du_{du}_hmm.pkl'
    with open(fname, 'rb') as fd:
        model = pickle.load(fd)

        return model


def model_dump(kinds, duration) -> dict:
    """
    Hidden Markov Model 을 반환함
    :param duration:
    :param kinds: load_dat
    :return:
    """
    model_dicts = OrderedDict()
    for chemical in kinds:
        print(f'Loading {chemical} HMM')
        hmm = load_hmm(chemical, duration)
        model_dicts[chemical] = hmm

    return model_dicts


def predict(model_dict: dict, test_dict: dict):
    """
    각 클래스별로 학습된 model을 이용해 각 클래스별로 생성된 test data를 대상으로 score(각 모델별 확률)를
    계산하고 이중 가장 score 값이 큰 클래스로 소속 판단

    :param model_dict: 분류 대상 클래스별 학습된 모델
    :param test_dict: 클래스별로 생성된 테스트 데이터
    :return:
    """
    predict_dict = OrderedDict()
    for chemical in test_dict:
        # chemical별 테스트 데이터 대상으로 모델 예측
        print('Predicting {}'.format(chemical))
        score_dict = {} # 모델별로 테스트 샘플을 대상으로 계산된 score 값 저장
        for hmm in model_dict:
            score = []
            for sample in test_dict[chemical]:
                # print(np.array(sample).reshape(1, -1).shape)
                temp_s = model_dict[hmm].score(np.array(sample).reshape(1, -1))
                score.append(temp_s)
            #score = [model_dict[hmm].score([sample]) for sample in test_dict[chemical]]
            score_dict[hmm] = score[:]
        temp_array = np.array(list(score_dict.values()))
        pred_array = temp_array.argmax(axis=0)
        predict_dict[chemical] = pred_array.tolist()

    return predict_dict


def metrics(num, predict_dict):
    actual = np.array([], dtype=np.int64)
    for i, chemical in enumerate(predict_dict):
        correct_answer = np.zeros(len(predict_dict[chemical])) + i
        actual = np.concatenate((actual, correct_answer))

    predicted = np.array([], dtype=np.int64)
    for value in list(predict_dict.values()):
        predicted = np.concatenate((predicted, np.array(value)))

    support = precision_recall_fscore_support(actual, predicted, average='micro')
    conf_mat = confusion_matrix(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    precision = {}
    recall = {}
    f1 = {}
    for i, chemical in enumerate(predict_dict):
        precision[chemical] = precision_score(actual, predicted, average=None, labels=[i])
        recall[chemical] = recall_score(actual, predicted, average=None, labels=[i])
        f1[chemical] = f1_score(actual, predicted, average=None, labels=[i])

    #print_metrics(num, conf_mat, accuracy, precision, recall, f1, support, list(predict_dict.keys()))

    return accuracy, recall, precision, f1


def print_metrics(num, conf_mat, accuracy, precision, recall, f1, support, chemicals):
    # if os.path.exists(os.path.join(droot, 'results/output.txt')):
    #     os.remove(os.path.join(droot, 'results/output.txt'))

    print(chemicals)
    out_file = '_'.join(chemicals)
    with open(os.path.join(droot, 'results/output_'+out_file+'_'+str(num)+'.txt'), 'a') as fd:
        title = ' vs. '.join(chemicals)
        fd.write('\n\n{}'.format(title))
        fd.write("\n==========================================================")
        fd.write('\nConfusion Matrix \n{}'.format(str(conf_mat)))
        fd.write("\nAccuracy: {}".format(str(accuracy)))

    print(title)
    print('Confusion Matrix')
    print(conf_mat)
    print("Accuracy")
    print(accuracy)
    print('\n')

    # print('results/output_' + out_file + '_' + str(num) + '.txt')
    # with open(os.path.join(droot, 'results/output_'+out_file+'_'+str(num)+'.txt'), 'a') as fd:
    #     for chemical in chemicals:
    #         fd.write('\nPrecision Score for {0}: {1:.3f}'.format(chemical, precision[chemical][0]))
    #         fd.write('\nRecall Score for {0}: {1:.3f}'.format(chemical, recall[chemical][0]))
    #         fd.write('\nF1 Score for {0}: {1:.3f}'.format(chemical, f1[chemical][0]))
    #     fd.write('\nSupport: Precision {0:.3f}, Recall {1:.3f}, F1 {2:.3f}'.format(*support))


    for chemical in chemicals:
        print('Precision Score for {}: {}'.format(chemical, precision[chemical][0]))
        print('Recall Score for {}: {}'.format(chemical, recall[chemical][0]))
        print('F1 Score for {}: {}'.format(chemical, f1[chemical][0]))
        print('\nSupport: Precision {0:.3f}, Recall {1:.3f}, F1 {2:.3f}'.format(*support))
    print('\nSupport: Precision {0:.3f}, Recall {1:.3f}, F1 {2:.3f}'.format(*support))


if __name__ == '__main__':
    pollutants = ['Benzen_0_1_ppm', 'Formaldehyde_0_1_ppm']
    train_x, train_y, test_x, test_y = load_data.timeseries_for_hmm(pollutants)
    # models_dict = model_training(train_x, save=True)
    # predict_dict = predict(models_dict, test_x)
    # accuracy, support = metrics(0, predict_dict)
    # print(accuracy)
    # print(support)

