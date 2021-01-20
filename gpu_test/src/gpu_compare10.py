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


def search_file(test, source, top_numb):
    print('\n%s --> %s' % (test, source))
    time_start_load_data = datetime.datetime.now()

    list_source = text_read(source)
    list_test = text_read(test)

    time_end_load_data = datetime.datetime.now()
    print('读文件用时：%s' % (time_end_load_data - time_start_load_data))

    results, result_frames = calculate_source_position_by_gpu(list_source, list_test, top_numb)
    # print('RESULT: %s  %s   %f  %d' % (test, source, results, result_frames))

    return results, result_frames


def calculate_source_position_by_gpu(list_source, list_test, top_numb):
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

    results, result_frames = find_most_probability(results, top_numb)
    for i in range(top_numb):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' %
              (i + 1, results[i]*100, result_frames[i], int(result_frames[i]/300),(result_frames[i]%300)/5))

    return results, result_frames


def find_most_probability(results, top_numb):
    top_results = np.array([0.0] * top_numb, dtype=np.float)
    top_results_frames = np.array([0] * top_numb, dtype=np.int)
    for i in range(len(results)):
        a = results[i]
        if a > top_results[0]:
            top_results_frames[0] = i + 1
            top_results[0] = a
            continue
        else:
            if a < top_results[-1]:
                continue
            else:
                for k in range(1, top_numb + 1):
                    if a > top_results[-k]:
                        continue
                    top_results_frames[-k + 1] = i + 1
                    top_results[-k + 1] = a
                    break
    return top_results, top_results_frames


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
    result_of_from_idx = accumulator / test_len / 4
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
        init_val += abs(3 - abs(src_frame_feature[1] - test_frame_feature[1]) * 5)
    if src_frame_feature[0] == test_frame_feature[2]:
        init_val += 1 - abs(src_frame_feature[1] - test_frame_feature[3])
    if src_frame_feature[2] == test_frame_feature[0]:
        init_val += 1 - abs(src_frame_feature[3] - test_frame_feature[1])
    if src_frame_feature[2] == test_frame_feature[2]:
        init_val += 1 - abs(src_frame_feature[3] - test_frame_feature[3])
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


def takefirst(elem):
    return elem[0]


if __name__ == '__main__':
    begin_load_cuda_time = datetime.datetime.now()
    print_gpu_info()
    end_load_cuda_time = datetime.datetime.now()
    print('CUDA driver init time: %s \n\n' % ( end_load_cuda_time - begin_load_cuda_time))

    start_time = datetime.datetime.now()
    top = 10
    if len(sys.argv) > 1:
        numb = int(sys.argv[1])
        for i in range(numb):
            test_file = sys.argv[2]
            source_file = sys.argv[3]
            final_result = []
            tmp_result = []
            tmp_result_1 = search_file(test_file, source_file, top)
            for i in range(top):
                a = (tmp_result_1[0][i], tmp_result_1[1][i], source_file)
                final_result.append(a)
                final_result.sort(key=takefirst, reverse=True)
                final_result = final_result[0:top:1]
            #print(final_result)
    else:
        source_files = get_files_from_dir('/home/ai/source')
        test_files = get_files_from_dir('/home/ai/test')
        for test_file in test_files:
            final_result = []
            tmp_result = []
            final_source_file = ''

            for source_file in source_files:
                tmp_result_1 = search_file('/home/ai/test/%s' % test_file, '/home/ai/source/%s' % source_file, top)
                for i in range(top):
                    a = (tmp_result_1[0][i], tmp_result_1[1][i], source_file)
                    final_result.append(a)
                final_result.sort(key=takefirst, reverse=True)
                final_result = final_result[0:top:1]

            print('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：' % test_file)
            for i in range(top):
                print('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' % (
                i + 1, final_result[i][0] * 100, final_result[i][2], final_result[i][1], int(final_result[i][1] / 300),
                (final_result[i][1] % 300) / 5))
            print('FResult: %s %.2f %% %d 分 %d 秒\n' % (final_result[0][2], final_result[0][0] * 100, int(final_result[0][1]/300), (final_result[0][1] % 300) / 5))

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
