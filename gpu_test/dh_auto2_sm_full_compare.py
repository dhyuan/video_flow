#!/usr/bin/env python
# coding:utf-8
import os, sys, getopt
import numpy as np
import tensorflow as tf
import time
import glob
import subprocess, sys
import socket
from datetime import datetime
from numba import cuda, jit
import math

# 准备目录
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(dir_path, 'saved_model')
model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')


def compare(d_test_features, d_sources_features_on_gpu, files_index, file_features_index_dict, value,
            threads_per_block=1024):
    compare_start_time = datetime.now()

    total_source_features_len = len(d_sources_features_on_gpu)
    total_blocks = math.ceil(total_source_features_len / threads_per_block)

    test_len = len(d_test_features)

    begin_block_index = 0
    begin_index_in_block = 0

    print('TotalSourceFeatures=%d, TotalBlocks=%d, test_len=%d' % (total_source_features_len, total_blocks, test_len))
    valid_block_t_indexes = np.empty((0, 5), dtype=np.int32)
    for file_index in range(len(file_features_index_dict)):
        feature_data = file_features_index_dict[file_index]
        begin_index = feature_data[0]
        end_index = feature_data[1]

        end_block_index = math.ceil((end_index - test_len) / threads_per_block)
        end_index_in_block = (end_index - test_len) % threads_per_block

        valid_block_t_indexes = np.vstack([valid_block_t_indexes,
                                           [file_index, begin_block_index, begin_index_in_block, end_block_index,
                                            end_index_in_block]])

        begin_block_index = math.ceil(end_index / threads_per_block)
        begin_index_in_block = end_index % threads_per_block

    d_valid_block_t_indexes = cuda.to_device(valid_block_t_indexes)
    cuda.synchronize()

    init_results = np.zeros((total_source_features_len, 3), dtype=np.float32)
    d_gpu_result = cuda.to_device(init_results)

    time_begin_gpu_cal = datetime.now()
    compare_frame_by_kernel[total_blocks, threads_per_block](total_source_features_len, test_len,
                                                             d_valid_block_t_indexes,
                                                             d_sources_features_on_gpu, d_test_features,
                                                             d_gpu_result)
    cuda.synchronize()
    h_results = np.zeros((total_source_features_len, 3), dtype=np.float32)
    d_gpu_result.copy_to_host(h_results)
    cuda.synchronize()

    time_end_gpu_cal = datetime.now()
    print('*** GPU计算用时：%s' % ((time_end_gpu_cal - time_begin_gpu_cal)))
    # print(h_results)

    ts_top10_result = find_most_probability(h_results, test_len, value)
    cpu_find_most_result_time = datetime.now()
    print('*** CPU cpu_find_most_result_time() 用时：%s' % (cpu_find_most_result_time - time_end_gpu_cal))

    final_result = []
    max_result = 0.0
    max_flag = 0

    if ts_top10_result is not None:
        if ts_top10_result[0][0] > max_result:
            max_result = ts_top10_result[0][0]
            max_flag = 1
        for i in range(len(ts_top10_result)):
            if max_result - ts_top10_result[i][0] < value:
                file_name = files_index[ts_top10_result[i][2]]
                one_result = (ts_top10_result[i][0], ts_top10_result[i][1], file_name)
                final_result.append(one_result)

    compare_end_time = datetime.now()
    print('*** CPU final_result 用时：%s' % (compare_end_time - cpu_find_most_result_time))

    if max_flag == 1:
        output_final_result = []
        for i in range(len(final_result)):
            if max_result - final_result[i][0] < value:
                one_result = (final_result[i][0], final_result[i][1], final_result[i][2])
                output_final_result.append(one_result)

        output_final_result.sort(key=take_first, reverse=True)
        print_final_top10(output_final_result, test_file)
        print('----------最大概率为：%.2f%%--------------------'%(max_result*100))
    else:
        print_final_top10(None, test_file)
        print('---------------最大概率小于15%，结果为空-------------------------------')

    end_time = datetime.now()
    print('在 %d 个文件中比对 %s 用时: %s' % (len(files_index), test_file, (end_time - compare_start_time)))


@cuda.jit
def compare_frame_by_kernel(total_source_features_len, test_len, d_valid_block_t_indexes,
                            d_src_frames_features, d_test_frames_features, d_results):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= total_source_features_len:
        return

    for i in range(len(d_valid_block_t_indexes)):
        index_scope = d_valid_block_t_indexes[i]
        file_index = index_scope[0]
        begin_block_idx = index_scope[1]
        begin_t_idx_in_block = index_scope[2]
        end_block_idx = index_scope[3]
        end_t_idx_in_block = index_scope[4]
        if begin_block_idx <= cuda.blockIdx.x <= end_block_idx and begin_t_idx_in_block <= cuda.threadIdx.x < end_t_idx_in_block:
            result_at_idx = calculate_features_probability_at_src_idx(idx, test_len, d_src_frames_features,
                                                                      d_test_frames_features)

            frame_index = idx - cuda.blockDim.x * begin_block_idx - begin_t_idx_in_block

            # 写回到全局内存
            cuda.atomic.add(d_results[idx], 0, result_at_idx)
            cuda.atomic.add(d_results[idx], 1, frame_index)
            cuda.atomic.add(d_results[idx], 2, file_index)

            #
        else:
            return

@cuda.jit(device=True)
def calculate_features_probability_at_src_idx(source_start_index, test_len,
                                              d_src_frames_features, d_test_frames_features):
    accumulator = 0
    for test_idx in range(test_len):
        src_idx = source_start_index + test_idx
        accumulator = compare_two_frame_feature(d_src_frames_features[src_idx], d_test_frames_features[test_idx],
                                                accumulator)
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


def log_data(file_name, src_features, h_results, source_len, source_features_len):
    print('RL file: %s  source_len=%d' % (file_name, source_len))
    # for i in range(source_len):
    #     print('\n%d  %f, %f, %f, %f' % (i, src_features[i][0], src_features[i][1], src_features[i][2], src_features[i][3]))
    #     print('%f' % (h_results[i]))


def print_final_top10(top10_result, file_name):
    print('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：' % file_name)
    (no_use, test_result_file_name) = os.path.split(file_name)
    (test_result_file_name_true, extension) = os.path.splitext(test_result_file_name)
    # f = open(result_dir + '/' + test_result_file_name_true + '_GPU_search_result.txt', 'a+')
    # f.write('\n\n-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：\n' % file_name)
    if top10_result != None:
        for i in range(len(top10_result)):
            print('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' % (
                i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
                (top10_result[i][1] % 300) / 5))
            # f.write('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处\n' % (
            #     i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
            #     (top10_result[i][1] % 300) / 5))
    else:
        print('---------------最大概率小于15%，结果为空-------------------------------')
        # f.write('None')
    # f.close()


def take_first(elem):
    return elem[0]


def print_top10(ts_top10_result, top_numb):
    for i in range(len(ts_top10_result)):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' %
              (i + 1, ts_top10_result[i][0] * 100, ts_top10_result[i][1], int(ts_top10_result[i][1] / 300),
               (ts_top10_result[i][1] % 300) / 5))


def find_most_probability_numpy(results, test_len, value):
    result_len = len(results)
    print('len of results %d' % result_len)
    tmp_results = np.empty((0, 3), dtype=np.float32)
    top_results = np.empty((0, 3), dtype=np.float32)

    max_result = 0.15
    max_frame = 0
    max_file_index = -1

    interval = test_len - 2
    max_flag = 0
    time1 = datetime.now()

    for i in range(result_len):
        time10 = datetime.now()
        frame_probability = results[i][0]
        frame_index = results[i][1] + 1
        file_index = results[i][2]
        time11 = datetime.now()
        # print("first loop --A1:  %s" % (time11 - time10))
        if frame_probability > max_result:
            max_result = frame_probability
            max_frame = frame_index
            max_file_index = file_index
            max_flag = 1
            continue
        if max_result - frame_probability < value and results[i][1] - max_frame > interval:
            tmp_results = np.vstack([tmp_results, [frame_probability, frame_index, file_index]])
            # tmp_results.append((frame_probability, frame_index, file_index))
        time19 = datetime.now()
        # print("first loop --A2:  %s" % (time19 - time11))
        # print("first loop --AA:  %s" % (time19 - time10))
    time2 = datetime.now()
    # print("first loop:  %s" % (time2 - time1))
    if max_flag == 1:
        max = (max_result, max_frame, max_file_index)
        top_results = np.vstack([top_results, [max_result, max_frame, max_file_index]])
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                top_results = np.vstack([top_results, [tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]]])
                # top_results.append((tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]))
        time3 = datetime.now()
        # print("first loop:  %s" % (time3 - time2))
        return (top_results)
    else:
        return (None)


def find_most_probability(results, test_len, value):
    result_len = len(results)
    print('len of results %d' % result_len)
    tmp_results = np.empty((0, 3), dtype=np.float32)
    top_results = np.empty((0, 3), dtype=np.float32)

    max_result = 0.15
    max_frame = 0
    max_file_index = -1

    interval = test_len - 2
    max_flag = 0
    time1 = datetime.now()

    results_max = results.max(axis=0)


    for i in range(result_len):
        time10 = datetime.now()
        frame_probability = results[i][0]
        frame_index = results[i][1] + 1
        file_index = results[i][2]
        time11 = datetime.now()
        # print("first loop --A1:  %s" % (time11 - time10))
        if frame_probability > max_result:
            max_result = frame_probability
            max_frame = frame_index
            max_file_index = file_index
            max_flag = 1
            continue
        if max_result - frame_probability < value and results[i][1] - max_frame > interval:
            tmp_results = np.vstack([tmp_results, [frame_probability, frame_index, file_index]])
            # tmp_results.append((frame_probability, frame_index, file_index))
        # time19 = datetime.now()
        # print("first loop --A2:  %s" % (time19 - time11))
        # print("first loop --AA:  %s" % (time19 - time10))

    # time2 = datetime.now()
    # print("first loop:  %s" % (time2 - time1))
    if max_flag == 1:
        max = (max_result, max_frame, max_file_index)
        top_results = np.vstack([top_results, [max_result, max_frame, max_file_index]])
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                top_results = np.vstack([top_results, [tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]]])
                # top_results.append((tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]))
        # time3 = datetime.now()
        # print("first loop:  %s" % (time3 - time2))
        return (top_results)
    else:
        return (None)


def trans_features_from_host2gpu(file_feature_dict):
    time_begin = time.time()

    mem_size = 0
    features_on_device_dict = {}
    for file_name, features in file_feature_dict.items():
        mem_size += len(features) * 4
        d_source_features = cuda.to_device(features)
        features_on_device_dict[file_name] = d_source_features

    time_end = time.time()
    print('特征值%d导入GPU用时：%d' % (mem_size, (time_end - time_begin)))
    return features_on_device_dict


def trans_all_source_features_as_whole(file_feature_dict):
    time_begin = time.time()

    total_features_numb = 0
    file_index_dict = {}
    file_features_index_dict = {}
    begin_index = 0
    file_index = 0
    print('prepare trans features')
    for file_name, features in file_feature_dict.items():
        source_features_numb = len(features)
        total_features_numb += source_features_numb

        file_index_dict[file_index] = file_name
        file_features_index_dict[file_index] = (begin_index, total_features_numb)

        begin_index = total_features_numb
        file_index += 1
    time_prepare_end = time.time()

    mem_size = total_features_numb * 4 * 4   # 点精度浮点32位，4字节，每个feature 16个字节。
    print('prepare trans features end. TotalFeatures: %d %d %s' % (total_features_numb, mem_size, (time_prepare_end - time_begin)))

    all_source_features = np.empty(shape=[0, 4])
    for file_id in range(len(file_features_index_dict.items())):
        file_name = file_index_dict[file_id]
        features = file_feature_dict[file_name]
        all_source_features = np.vstack((all_source_features, features))
    time_prepare_end2 = time.time()
    print('prepare all_source_features end. %s' % (time_prepare_end2 - time_prepare_end))

    d_all_source_features = cuda.to_device(all_source_features)
    time_end = time.time()


    print('to_device time. %s' % (time_end - time_prepare_end2))
    print('特征值%d导入GPU用时：%d' % (mem_size, (time_end - time_begin)))
    return d_all_source_features, file_index_dict, file_features_index_dict


def read_sources_features_from_dir(features_dir, fun_features_str2tuple):
    time_begin = datetime.now()
    files = get_files_from_dir(features_dir)
    file_feature_dict = {}
    counter = 0
    for file in files:
        if os.path.isdir('%s/%s' % (features_dir, file)) or file.startswith('.') or os.path.getsize(
                '%s/%s' % (features_dir, file)) == 0:
            print('忽略空文件: %s' % file)
            continue

        source_file = '%s/%s' % (features_dir, file)
        list_of_features = text_read(source_file)
        counter += 1
        print('src-file %d  %s %d' % (counter, source_file, len(list_of_features)))

        source_features = np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float32)
        file_feature_dict[file] = source_features
    time_end = datetime.now()
    print('读取%d个源特征文件用时：%s' % (len(files), (time_end - time_begin)))
    return file_feature_dict


def load_features_to_gpu(sources_dir):
    file_feature_dict = read_sources_features_from_dir(sources_dir, trans_source_data_to_tuple)
    d_features_on_device_dict = trans_features_from_host2gpu(file_feature_dict)
    return d_features_on_device_dict, file_feature_dict


def read_test_features_from_file(test_file, fun_features_str2tuple):
    list_of_features = text_read(test_file)
    test_features = np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float)
    return test_features


def load_test_features_to_gpu(features):
    d_test_features = cuda.to_device(features)
    return d_test_features


def text_read(f):
    try:
        lines = open(f, 'r').readlines()
        return lines
    except:
        print('ERROR, 结果文件不存在！')


def trans_source_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def trans_test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    return files


if __name__ == '__main__':
    value = 0.05
    sources_dir = './data/source'  # 比对 所有源的特征值文本 存放路径
    test_dir = './data/test'

    test_files = get_files_from_dir(test_dir)

    file_feature_dict = read_sources_features_from_dir(sources_dir, trans_source_data_to_tuple)
    d_all_sources_features, file_index_dict, file_features_index_dict = trans_all_source_features_as_whole(
        file_feature_dict)

    for test_file in test_files:
        test_features = read_test_features_from_file('./data/test/%s' % test_file, trans_test_data_to_tuple)
        d_test_features = load_test_features_to_gpu(test_features)

        compare(d_test_features, d_all_sources_features, file_index_dict, file_features_index_dict, value)
