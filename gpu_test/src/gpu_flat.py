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

err_value_numb = 0

def calculate_source_position_by_gpu(list_source, list_test):
    source_features = np.array(list(map(lambda d: trans_source_data_to_tuple(d), list_source)), dtype=np.float)
    test_features = np.array(list(map(lambda d: trans_test_data_to_tuple(d), list_test)), dtype=np.float)
    source_features_len = len(source_features)
    test_features_len = len(test_features)
    source_len = source_features_len - test_features_len
    test_len = test_features_len
    print('source_features_len=%d source_len=%d test_len=%d' % (source_features_len, source_len, test_len))

    threads_per_block = 512  # TODO: 应从设备信息获取
    blocks_per_grid = math.ceil(source_len / threads_per_block)
    print('threads_per_block=%d  blocks_per_grid=%d' % (threads_per_block, blocks_per_grid))

    # copy memory from Host to Device
    time_start_h2d = datetime.datetime.now()
    d_source_features = cuda.to_device(source_features)
    d_test_features = cuda.to_device(test_features)
    d_gpu_result = cuda.device_array(source_len)
    time_end_h2d = datetime.datetime.now()
    cuda.synchronize()
    print('内存copy用时：%s' % (time_end_h2d - time_start_h2d))

    compare_frame_by_kernel[blocks_per_grid, threads_per_block](source_len, test_len,
                                                                d_source_features, d_test_features,
                                                                d_gpu_result)
    cuda.synchronize()
    time_end_gpu_calculation = datetime.datetime.now()
    print('*** GPU计算用时：%s' % (time_end_gpu_calculation - time_end_h2d))

    results = np.array([0.0] * source_len, dtype=np.float)
    d_gpu_result.copy_to_host(results)
    cuda.synchronize()

    result, result_frame = find_most_probability(results)
    time_end_cal = datetime.datetime.now()
    if result > 100:
        #err_value_numb = err_value_numb + 1
        print('EEEE')
    print('匹配度概率为：%.2f%%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分 %d 秒处'%(result*100,result_frame,int(result_frame/300),(result_frame%300)/5))
    print('*** 匹配用时：%s' % (time_end_cal - time_end_h2d))
    
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

    source_frame_start_index = idx
    #result_of_from_idx = calculate_features_probability_at_src_idx(source_frame_start_index, test_len, d_src_frames_features, d_test_frames_features)
    accumulator = 0.0
    init_val = 0.0
    for test_idx in range(test_len):
        src_idx = source_frame_start_index + test_idx
        #accumulator = compare_two_frame_feature(d_src_frames_features[src_idx], d_test_frames_features[test_idx], accumulator)
        d_src_frame_feature = d_src_frames_features[src_idx]
        d_test_frame_feature = d_test_frames_features[test_idx]
        if d_src_frame_feature[0] == d_test_frame_feature[0]:
            init_val = init_val + 3 - abs((d_src_frame_feature[1] - d_test_frame_feature[1])) * 9
        if d_src_frame_feature[0] == d_test_frame_feature[2]:
            init_val = init_val + 1 - abs((d_src_frame_feature[1] - d_test_frame_feature[3]))
        if d_src_frame_feature[2] == d_test_frame_feature[0]:
            init_val = init_val + 1 - abs((d_src_frame_feature[3] - d_test_frame_feature[1]))
        if d_src_frame_feature[2] == d_test_frame_feature[2]:
            init_val = init_val + 1 - abs((d_src_frame_feature[3] - d_test_frame_feature[3]))
    accumulator = init_val
    result_of_from_idx = accumulator / test_len / 8
    #--------

    cuda.atomic.add(d_results, source_frame_start_index, result_of_from_idx)


@cuda.jit(device=True)
def calculate_features_probability_at_src_idx(source_frame_start_index, test_len,
                                              d_src_frames_features, d_test_frames_features):
    accumulator = 0
    for test_idx in range(test_len):
        src_idx = source_frame_start_index + test_idx
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
    load_cuda[2, 2](d_init_data)



def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    return files


if __name__ == '__main__':
    begin_load_cuda_time = datetime.datetime.now()
    print_gpu_info()
    end_load_cuda_time = datetime.datetime.now()
    print('CUDA driver init time: %s \n\n' % ( end_load_cuda_time - begin_load_cuda_time))

    start_time = datetime.datetime.now()
    source_files = get_files_from_dir('/home/ai/source')
    test_files = get_files_from_dir('/home/ai/test')
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
        test_file = sys.argv[2]
        search_file(test_file, source_file)
    else:
        for source_file in source_files:
            for test_file in test_files:
                search_file('/home/ai/test/%s' % test_file, '/home/ai/source/%s' % source_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
    #print("erro_value_numb=%d" % err_value_numb)


