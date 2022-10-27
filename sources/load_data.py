import os
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd

from tensorflow import keras

target = {'Normal': 0,
          'Benzen_0_1_ppm': 1, 'Benzen_0_5_ppm': 2,
          'Formaldehyde_0_1_ppm': 3, 'Formaldehyde_0_5_ppm': 4,
          'Toluen_0_1_ppm': 5, 'Toluen_0_5_ppm': 6}
data_path = os.path.join(pathlib.Path(__file__).parents[1], 'data')
# '../../data/'
train_rate = 0.6 # the ratio of training data
valid_rate = 0.2 # the ratio of validation data
test_rate = 0.2 # the ratio of test data (== remaining data except train data + valid data


def timeseries_dataset(pollutants: list,
                       start=10,
                       end=40,
                       duration=30,
                       batch_size=128) -> '(train_set, valid_set, test_set, samples)':
    """
    오염 물질들의 데이터 파일 전체를 keras.utils.timeseries_dataset_from_array를 이용해 time series data_sets set 으로 변환

    :param pollutants: list
        ex) ['Formaldehyde_0_1_ppm', 'Normal']
    :param start: int
        The default is 10(분). 60분 짜리 csv 파일에서 실제 데이터로 사용할 부분의 시작 시각
    :param end: int
        The default is 40(분). 60분 짜리 csv 파일에서 실제 데이터로 사용할 부분의 마지막 시각
    :param duration: int
        The default is 30(초). 학습 및 판별에 사용할 관찰 시간
    :param batch_size: int
        The default is 128. batch size

    :return: tuple of BatchDataset
        (train_dataset, valid_dataset, test_dataset)
    """

    data = defaultdict(list)  # data points:
    classes = defaultdict(list)  # target
    num_frames = dict()  # the number of frames for each pollutant
    num_samples = dict()  # the number of data_sets points with the size of sequence length for each pollutant
    start_frame = start * 60 * 4  # start(min) x 60(sec) x 4(frame per second)
    end_frame = end * 60 * 4
    sequence_length = duration * 4

    print(pollutants)
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

    # data_sets = np.float32(np.concatenate(data_sets, axis=0)) # for SOM
    train_x = []; train_y = []
    valid_x = []; valid_y = []
    test_x = []; test_y = []

    for key in data.keys():
        n_train = int(len(data[key]) * train_rate) # the number of training data samples
        n_valid = int(len(data[key]) * valid_rate) # the number of validation data samples
        train_x.append(data[key][:n_train])
        train_y.append(classes[key][:n_train])
        valid_x.append(data[key][n_train:n_train+n_valid])
        valid_y.append(classes[key][n_train:n_train + n_valid])
        test_x.append(data[key][n_train + n_valid:])
        test_y.append(classes[key][n_train + n_valid:])

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    valid_x = np.concatenate(valid_x, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)


    ## List comprehension version for data set ##
    # train_x = [data[x][:] for x in data.keys()]
    # train_x = np.concatenate(train_x, axis=0)
    # train_y = [classes[x][:int(len(classes[x]) * tr_rate)] for x in classes.keys()]
    # train_y = np.concatenate(train_y, axis=0)
    # valid_x = [data[x][int(len(data[x]) * tr_rate):int(len(data[x])*tr_rate)+int(len(data[x])*va_rate)] for x in data.keys()]
    # valid_x = np.concatenate(valid_x, axis=0)
    # valid_y = [classes[x][int(len(data[x]) * tr_rate):int(len(data[x])*tr_rate)+int(len(data[x])*va_rate)] for x in classes.keys()]
    # valid_y = np.concatenate(valid_y, axis=0)
    # test_x = [data[x][int(len(data[x])*tr_rate)+int(len(data[x])*va_rate):] for x in data.keys()]
    # test_x = np.concatenate(test_x, axis=0)
    # test_y = [classes[x][int(len(data[x])*tr_rate)+int(len(data[x])*va_rate):] for x in classes.keys()]
    # test_y = np.concatenate(test_y, axis=0)

    # return data, classes, train_x, train_y, valid_x, valid_y, test_x, test_y

    # num_train_set = int(data_sets.shape[0] * train_rate)
    # num_valid_set = int(data_sets.shape[0] * valid_rate)
    # num_test_set = data_sets.shape[0] - (num_train_set + num_valid_set)
    #
    # print(f'Pollutant: {pollutant}')
    # print(f'the number of frames: num_train_set {num_train_set}, num_valid_set {num_valid_set}, num_test_set {num_test_set}')
    #
    # sequence_length = duration * 4
    # targets = np.full(len(data_sets), target[pollutant])
    #
    train_dataset = keras.utils.timeseries_dataset_from_array(
        train_x,
        targets=train_y,
        sequence_length=0,
        shuffle=True,
        batch_size=batch_size,
        start_index=0
    )

    # valid_dataset = keras.utils.timeseries_dataset_from_array(
    #     data_sets,
    #     targets=targets,
    #     sequence_length=sequence_length,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     start_index=num_train_set,
    #     end_index=num_train_set+num_valid_set
    # )
    #
    # test_dataset = keras.utils.timeseries_dataset_from_array(
    #     data_sets,
    #     targets=targets,
    #     sequence_length=sequence_length,
    #     shuffle=True,
    #     batch_size=batch_size,
    #     start_index=num_train_set + num_valid_set
    # )
    #

    return train_dataset#, valid_dataset, test_dataset


if __name__ == '__main__':
    train_dataset = timeseries_dataset(['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm'])
    # data, classes, train_x, train_y, valid_x, valid_y, test_x, test_y = timeseries_dataset(['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm'])

    # # 데이터 크기 확인용
    # if train_dataset:
    #     num_data = 0
    #     for _, t in train_dataset:
    #         num_data += len(t)
    #     print(f'the size of train data set: {num_data}')
    #     # print(num_data, int(len(samples) * valid_rate) - 120 + 1)
    # if valid_dataset:
    #     num_data = 0
    #     for _, t in valid_dataset:
    #         num_data += len(t)
    #     print(f'the size of validation data set: {num_data}')
    # if test_dataset:
    #     num_data = 0
    #     for _, t in test_dataset:
    #         num_data += len(t)
    #     print(f'the size of test data set: {num_data}')