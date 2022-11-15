import os, sys
import pathlib
import shutil
import pandas as pd
import numpy as np
import somoclu
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_classification
# from somlearn import SOM

data_root = '/home/drminor/PycharmProjects/elegans/data/'
directory = {'Form_01': 'Formaldehyde_0_1_ppm',
                 'Form_05': 'Formaldehyde_0_5_ppm',
                 'Normal': 'Normal',
                 'Benzen_01': 'Benzen_0_1_ppm',
                 'Benzen_05': 'Benzen_0_5_ppm',
                 'Toluen_01': 'Toluen_0_1_ppm',
                 'Toluen_05': 'Toluen_0_5_ppm'}
nfiles = []  # the number of files(=samples * 2) for each directory, 2 files per sample
frames_per_min = 60 * 4


def load_data(pollutants, start=10, end=40):
    """
    load frames from each directory in kinds for SOM analysis

    Arguments
        kinds: tuple of strings, optional
            The default is ('Form_01', 'Form_05', 'Normal')
        start: int, optional
            데이터의 시작 위치 - 분 단위
            The default is 10 - 10 * 60(sec) * 4(frames) = 2400
        end: int, optional
            데이터의 마지막 위치 - 분 단위
            The default is 30 - 30 * 60(sec) * 4(frames) = 7200

    Returns
        frames: 2-dim np.ndarray
    """

    global nfiles
    nfiles = dict()
    frames = []

    for chemical in pollutants:
        print('Loading: {}'.format(chemical))
        files = os.listdir(os.path.join(data_root, chemical))
        files.sort()
        nfiles[chemical] = len(files)  # the number of files in each directory

        for file in files:
            if file.endswith('csv'):
                # print(file)
                df = pd.read_csv(os.path.join(data_root, chemical, file), header=None)
                df.dropna(axis=0, inplace=True)
                #print(file, df.shape)
                frames.append(df.values[start * frames_per_min:end * frames_per_min])  # 4800 frames per sample (after 10 mins and interval of 20 mins)

    frames = np.float32(np.concatenate(frames))

    return frames, nfiles # somoclue requires float32 dtype


def load_all_data(pollutants, start=10, end=30):
    """
    load frames from each directory in kinds for SOM analysis

    Arguments
        kinds: tuple of strings, optional
            The default is ('Form_01', 'Form_05', 'Normal')
        start: int, optional
            데이터의 시작 위치 - 분 단위
            The default is 10 - 10 * 60(sec) * 4(frames) = 2400
        end: int, optional
            데이터의 마지막 위치 - 분 단위
            The default is 30 - 30 * 60(sec) * 4(frames) = 7200

    Returns
        frames: 2-dim np.ndarray
    """

    global nfiles
    nfiles = dict()
    data_root = pathlib.Path(__file__).parents[1]
    frames = []

    for chemical in kinds:
        print('Loading: {}'.format(chemical))
        direct = directory[chemical]
        files = os.listdir(os.path.join(data_root, direct))
        files.sort()
        nfiles[chemical] = len(files)  # the number of files in each directory

        for file in files:
            if file.endswith('csv'):
                df = pd.read_csv(os.path.join(data_root, direct, file), header=None)
                df.dropna(axis=0, inplace=True)
                print(file, df.shape)
                frames.append(df.values[start * frames_per_min:end * frames_per_min])  # 4800 frames per sample (after 10 mins and interval of 20 mins)

    print(nfiles)
    frames = np.float32(np.concatenate(frames))

    return frames # somoclue requires float32 dtype


def som_analysis(frames, n_columns=14, n_rows=14, n_cluster=7, min_clusters=3, max_clusters=10):
    """
    SOM 및 Kmeans clustering 분석을 통해 패펀을 결정하고 데이터를 패턴 시퀀스로 변경

    :param frames: np.ndarray (np.float32)
    :param n_columns: int, optional
    :param n_rows:
    :param min_clusters:
    :param max_clusters:
    :param start:
    :param end:
    :return:
    """
    # kinds = ('Form_01', 'Form_05', 'Normal')
    #
    # frames = load_all_data(kinds=kinds, start=start, end=end)
    scaler = MinMaxScaler()
    scaled_frames = scaler.fit_transform(frames)

    print("SOM analysing --")
    som = somoclu.Somoclu(n_columns=n_columns, n_rows=n_rows, compactsupport=False)
    som.train(scaled_frames)
    codebook = som.codebook  # codebook is the weights of each node, codebook.shape : (n_rows, n_columns, dim_features)
    flat_codebook = codebook.reshape(n_columns * n_rows,
                                     frames.shape[-1])  # for KMeans input - [0][0]:0, [0][1]:1, ..., [1][0]:[14] ...
    # print(type(flat_codebook))
    # print(flat_codebook.shape)
    # print(flat_codebook)
    # print(type(som.bmus))
    # print(som.bmus.shape)
    # print('data: {}'.format(frames.shape))
    # print(som.bmus)
    # 최적 클러시스터링을 찾기 위함
    # - KMeans와 davies-boulding index는 k 개의 패턴을 찾는데 불안정
    # - 다양한 클러스터링 방법과 메트릭을 활용해 분류 성능 분석 필요
    # - SOM 맵의 구조 toroid?, planar, emergent map

    # min = 20.
    # best_kmeans = None
    # best_ncluster = -1
    # for clusters in range(min_clusters, max_clusters+1):
    #     kmeans = KMeans(n_clusters=clusters, random_state=42).fit(flat_codebook)
    #     score = davies_bouldin_score(flat_codebook, kmeans.labels_)
    #     print(score)
    #     if score < min:
    #         min = score
    #         best_ncluster = clusters
    #         best_kmeans = kmeans
    # print(best_ncluster, ': ', min, end='\n')
    # print(best_kmeans.labels_)

    print("K Means clustering --")
    best_kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(flat_codebook)  # codebook clustering

    # 2차원으로 된 bmu를 codebook 클러스터링 후 1차원 bmu로 변환 -> 패턴 시퀀스 생성
    temporal_patterns = []
    for i, j in som.bmus:
        temporal_patterns.append(best_kmeans.labels_[i * n_rows + j])

    return np.array(temporal_patterns, dtype=np.int32)


def divide_sequences(sequences, polls, nfiles, start=10, end=30):
    """
    full sequences를 화학물질 별로 분할

    :param sequences: np.ndarray (1-dim)
        전체 데이터의 패턴 시퀀스
    :param specimen: dictinary
        화학물질 별 선택된 샘플들
    :param kinds: tuple of string
        화학 물질의 종류
    :param start: int, optional
        데이터 시작 시간, 최초 시작으로 부터 start 분
    :param end:
        데이터 종료 시간, 최초 시작으로 부터 end 분

    :return pattern-dict: dictionary of np.ndarray
        화학 물질 별 패턴 시퀀스
    """
    pattern_dict = {}
    nframes = (end - start) * frames_per_min # 주어진 시간(분: ex-20분)동안의 샘플 개체별 데이터 프레임 전체
    accu_nfiles = np.cumsum(np.asarray(list(nfiles.values())))
    print(polls)

    for i, kind in enumerate(polls):
        print(i, kind)
        if i == 0:
            pattern_dict[kind] = sequences[:accu_nfiles[i] * nframes]
        else:
            pattern_dict[kind] = sequences[accu_nfiles[i - 1] * nframes:accu_nfiles[i] * nframes]

    return pattern_dict


def transform_input_data(seq_dict, tinter=20, overlap=0.3, save=True):
    """
    HMM에서 사용 가능한 입력 데이터로 변형

    :param seq_dict: dictionary
        화학 물질별 패턴 시퀀스
    :param tinter: int, optional
        모델 학습 및 테스트에 사용하기 위한 관찰 간격(초) - 20 * 4 = 80 frames
        The default is 20.
    :param overlap: float, optional
        gap effect를 제거하기 위해 관찰 시퀀스간 중복되는 비율
    :return input_data: dictionary of list of list
    """

    len_input_seq = tinter * 4 # ex) 80
    len_overlap = int(len_input_seq * overlap) # 80 * 0.3 = 24
    net_inter = len_input_seq - len_overlap # 80 - 24 = 56
    input_data = dict()

    print('Transforming sequences to input data')
    for chemical in seq_dict:
        seq = seq_dict[chemical]
        seq_list = []
        if len(seq) % net_inter >= len_overlap:
            n_seq = int(len(seq)/net_inter)
        else:
            n_seq = int(len(seq)/net_inter) - 1
        for i in range(n_seq):
            temp = seq[i*net_inter : i*net_inter+len_input_seq]
            seq_list.append(temp.tolist())
        input_data[chemical] = seq_list

    if save:
        print('Saving input data')
        for chemical in input_data:
            fpath = os.path.join(pathlib.Path(__file__).parents[0], 'results',
                                 '{}_{}_{}.csv'.format(chemical, len_input_seq, len_overlap))
            df = pd.DataFrame(input_data[chemical], dtype=np.int32)
            df.to_csv(fpath, index=False, header=False)

    return input_data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pollutants = ['Formaldehyde_0_1_ppm', 'Benzen_0_1_ppm']

    start = 10
    end = 40
    tinter = 120
    n_cluster = 5
    n_component = 5

    data, nfiles = load_data(pollutants, start=start, end=end)
    seqs = som_analysis(data, n_columns=14, n_rows=14, n_cluster=n_cluster)
    # divided_seq = divide_sequences(seqs, specimen, *kinds, start=start, end=end)

    # kinds = ('Form_01', 'Toluen_01')
    # frames = load_all_data(*kinds)
    # print(nfiles)
    # best = som_analysis(frames)

    # frames = load_all_data()
    # nrows, ncols = 14, 14
    # scaler = MinMaxScaler()
    # scaled_frames = scaler.fit_transform(frames)
    # som = somoclu.Somoclu(n_columns=14, n_rows=14, maptype='toroid', compactsupport=False)
    # som.train(scaled_frames)

    # file_list = search_files(type='py')
    # aggregate_pythons(file_list)
    # file_list = search_files(type='m')
    # aggregate_matlab(file_list)
    # X, _ = make_classification(random_state=0)
    # som = SOM(n_columns=3, n_rows=3, random_state=1)
    # som.fit(X)
    # file_rename()