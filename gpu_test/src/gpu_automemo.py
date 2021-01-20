import datetime
import sys
from numba import cuda, jit
import numpy as np
import math
import os


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def trans_source_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def trans_test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def search_file(test, source):
    print('\n%s --> %s' % (test, source))
    time_start_load_data = datetime.datetime.now()

    list_source = text_read(source)
    list_test = text_read(test)

    time_end_load_data = datetime.datetime.now()
    print('\n读文件用时：%s' % (time_end_load_data - time_start_load_data))

    result, result_frame = calculate_source_position_by_gpu(list_source, list_test)
    return result, result_frame


def calculate_source_position_by_gpu(list_source, list_test):
    source_features = np.array(list(map(lambda d: trans_source_data_to_tuple(d), list_source)), dtype=np.float)
    test_features = np.array(list(map(lambda d: trans_test_data_to_tuple(d), list_test)), dtype=np.float)

    source_features_len = len(source_features)
    test_features_len = len(test_features)

    source_len = source_features_len - test_features_len
    test_len = test_features_len
    print('source_features_len=%d source_len=%d test_len=%d' % (source_features_len, source_len, test_len))

    threads_per_block = 1024  # TODO: 应从设备信息获取
    blocks_per_grid = math.ceil(source_len / threads_per_block)
    print('threads_per_block=%d  blocks_per_grid=%d' % (threads_per_block, blocks_per_grid))

    time_begin_gpu_calculation = datetime.datetime.now()
    results = np.array([0.0] * source_len, dtype=np.float)
    compare_frame_by_kernel[blocks_per_grid, threads_per_block](source_len, test_len,
                                                                source_features, test_features,
                                                                results)
    cuda.synchronize()
    time_end_gpu_calculation = datetime.datetime.now()
    print('*** GPU计算用时：%s' % (time_end_gpu_calculation - time_begin_gpu_calculation))

    result, result_frame = find_most_probability(results)
    print('匹配度概率为：%.2f%%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分 %d 秒处'%(result*100,result_frame,int(result_frame/300),(result_frame%300)/5))
    if result > 100:
        print('ERROR')
    return result, result_frame


def find_most_probability(results):
    result = 0
    result_frame = 0
    for i in range(len(results)):
        if results[i] > result:
            result_frame = i + 1
            result = results[i]
    return result, result_frame


@cuda.jit
def compare_frame_by_kernel(source_len, test_len,
                            d_src_frames_features, d_test_frames_features,
                            d_results):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= source_len:
        return

    src_frame_start_index = idx
    result_of_from_idx = calculate_features_probability_at_src_idx(src_frame_start_index, test_len,
                                                                   d_src_frames_features, d_test_frames_features)
    cuda.atomic.add(d_results, src_frame_start_index, result_of_from_idx)


@cuda.jit(device=True)
def calculate_features_probability_at_src_idx(source_start_index, test_len,
                                              d_src_frames_features, d_test_frames_features):
    accumulator = 0
    for test_idx in range(test_len):
        src_idx = source_start_index + test_idx
        accumulator = compare_two_frame_feature(d_src_frames_features[src_idx], d_test_frames_features[test_idx], accumulator)
    result_of_from_idx = accumulator / test_len / 8
    return result_of_from_idx


@cuda.jit(device=True)
def compare_two_frame_feature(src_frame_feature, test_frame_feature, init_val):
    """
    计算两帧特征的相似度。
    :param src_frame_feature: 原帧的特征数据。(Top features with possibility. Tuple of float.)
    :param test_frame_feature: 待检测帧的特征数据。
    :param init_val: 初始值。
    :return: 比较结果
    """
    if src_frame_feature[0] == test_frame_feature[0]:
        init_val += 3 - abs((src_frame_feature[1] - test_frame_feature[1])) * 9
    if src_frame_feature[0] == test_frame_feature[2]:
        init_val += 1 - abs((src_frame_feature[1] - test_frame_feature[3]))
    if src_frame_feature[2] == test_frame_feature[0]:
        init_val += 1 - abs((src_frame_feature[3] - test_frame_feature[1]))
    if src_frame_feature[2] == test_frame_feature[2]:
        init_val += 1 - abs((src_frame_feature[3] - test_frame_feature[3]))
    return init_val


@cuda.jit
def load_cuda(d_init_data):
    pass


def print_gpu_info():
    print(cuda.gpus)
    init_data = np.array([1, 2])
    d_init_data = cuda.to_device(init_data)
    load_cuda[2, 128](d_init_data)


def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    return files


if __name__ == '__main__':
    begin_load_cuda_time = datetime.datetime.now()
    print_gpu_info()
    end_load_cuda_time = datetime.datetime.now()
    print('CUDA driver init time: %s \n\n' % ( end_load_cuda_time - begin_load_cuda_time))

    start_time = datetime.datetime.now()
    if len(sys.argv) > 1:
        numb = int(sys.argv[1])
        for i in range(numb):
            source_file = sys.argv[2]
            test_file = sys.argv[3]
            search_file(test_file, source_file)
    else:
        i = 0
        source_files = get_files_from_dir('/home/ai/source')
        test_files = get_files_from_dir('/home/ai/test')
        for source_file in source_files:
            for test_file in test_files:
                search_file('/home/ai/test/%s' % test_file, '/home/ai/source/%s' % source_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
