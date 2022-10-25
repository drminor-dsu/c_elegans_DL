import os
import pathlib
import numpy as np
import pandas as pd

from tensorflow import keras

target = {'Normal': 0,
        'Benzen_0_1_ppm': 1, 'Benzen_0_5_ppm': 2,
        'Formaldehyde_0_1_ppm': 3, 'Formaldehyde_0_5_ppm': 4,
        'Toluen_0_1_ppm': 5, 'Toluen_0_5_ppm': 6}
data_path = os.path.join(pathlib.Path(__file__).parents[1], 'data')
# '../../data/'
train_rate = 0.6
valid_rate = 0.2
test_rate = 0.2

def timeseries_dataset(pollutant:str,
                       start=10,
                       end=40,
                       duration=30,
                       batch_size=128)->'(train_set, valid_set, test_set, samples)':
    """
    특정 오염 물질 데이터 파일 전체를 keras.utils.timeseries_dataset_from_array를 이용해 time series data set 으로 변환

    :param pollution: string
        ex) 'Formaldehyde_0_1_ppm'
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

    data = []
    num_frames = 0
    start_frame = start * 60 * 4 # start(min) x 60(sec) x 4(frame per second)
    end_frame = end * 60 * 4
    directory = os.path.join(data_path, pollutant)
    files = os.listdir(directory)
    files.sort()
    for csvfile in files:
        #print(csvfile)
        df = pd.read_csv(os.path.join(directory, csvfile), header=None)
        df.dropna(axis=0, inplace=True)
        df = df[start_frame: end_frame]
        num_frames += df.shape[0]
        #print(df.shape)
        data.append(df.values)
    samples = np.concatenate(data, axis=0)
    # samples = np.float32(np.concatenate(data, axis=0)) # for SOM

    num_train_set = int(samples.shape[0] * train_rate)
    num_valid_set = int(samples.shape[0] * valid_rate)
    num_test_set = samples.shape[0] - (num_train_set + num_valid_set)

    print(f'Pollutant: {pollutant}')
    print(f'the number of frames: num_train_set {num_train_set}, num_valid_set {num_valid_set}, num_test_set {num_test_set}')

    sequence_length = duration * 4
    targets = np.full(len(samples), target[pollutant])

    train_dataset = keras.utils.timeseries_dataset_from_array(
        samples,
        targets=targets,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_set
    )

    valid_dataset = keras.utils.timeseries_dataset_from_array(
        samples,
        targets=targets,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_set,
        end_index=num_train_set+num_valid_set
    )

    test_dataset = keras.utils.timeseries_dataset_from_array(
        samples,
        targets=targets,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=num_train_set + num_valid_set
    )

    return train_dataset, valid_dataset, test_dataset, samples


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, _ = timeseries_dataset('Formaldehyde_0_1_ppm')

    # 데이터 크기 확인용
    if train_dataset:
        num_data = 0
        for _, t in train_dataset:
            num_data += len(t)
        print(f'the size of train data set: {num_data}')
        # print(num_data, int(len(samples) * valid_rate) - 120 + 1)
    if valid_dataset:
        num_data = 0
        for _, t in valid_dataset:
            num_data += len(t)
        print(f'the size of validation data set: {num_data}')
    if test_dataset:
        num_data = 0
        for _, t in test_dataset:
            num_data += len(t)
        print(f'the size of test data set: {num_data}')