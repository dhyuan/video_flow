#!/usr/bin/env python
#coding:utf-8
import os, sys, getopt
import numpy as np
import tensorflow as tf
import time
import glob
import subprocess,sys
import socket
from datetime import datetime
from numba import cuda, jit
import math


# 准备目录
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(dir_path, 'saved_model')
model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')


def compare(test_file, d_sources_features_on_gpu, file_feature_dict, value):
    start_time = datetime.now()

    sources_numb = len(d_sources_features_on_gpu)
    end_load_source_time = datetime.now()

    test_features = read_test_features_from_file(test_file, trans_test_data_to_tuple)
    d_test_features = load_test_features_to_gpu(test_features)

    test_len = len(test_features)
    threads_per_block = 1024

    begin_compare_time = datetime.now()
    final_result = []
    max_result = 0.0
    max_flag = 0

    for file_name, d_source_features in d_sources_features_on_gpu.items():
        # print('%s -- %s' % (test_file, file_name))
        source_features_len = len(d_source_features)
        source_len = source_features_len - test_len

        blocks_per_grid = math.ceil(source_len / threads_per_block)
        # print('threads_per_block=%d  blocks_per_grid=%d' % (threads_per_block, blocks_per_grid))

        init_results = np.array([0.0] * source_len, dtype=np.float)
        d_gpu_result = cuda.to_device(init_results)
        #d_gpu_result = cuda.device_array(source_len)
        cuda.synchronize()

        last_idx = source_features_len - 1
        # print('%f, %f, %f, %f' % (d_source_features[0][0], d_source_features[0][1], d_source_features[0][2], d_source_features[0][3]))
        # print('%f, %f, %f, %f' % (d_source_features[last_idx][0], d_source_features[last_idx][1], d_source_features[last_idx][2], d_source_features[last_idx][3]))
        time_begin_gpu_cal = datetime.now()
        # print('--source_features_len=%d source_len=%d  test_len=%d' % (source_features_len, source_len, test_len))
        compare_frame_by_kernel[blocks_per_grid, threads_per_block](source_len, test_len,
                                                                    d_source_features, d_test_features,
                                                                    d_gpu_result)
        h_results = np.array([0.0] * source_len, dtype=np.float)
        d_gpu_result.copy_to_host(h_results)
        cuda.synchronize()

        # print('---h_results size %d' % len(h_results))
        time_begin_cpu_cal = datetime.now()
        ts_top10_result = find_most_probability(h_results, test_len,value)
        time_end_cpu_cal = datetime.now()
        # print('CPU time %s' % (time_end_cpu_cal - time_begin_cpu_cal))
        # print_top10(ts_top10_result, value)

        if ts_top10_result != None:
            if ts_top10_result[0][0] > max_result:
                max_result = ts_top10_result[0][0]
                max_flag = 1
            for i in range(len(ts_top10_result)):
                if max_result - ts_top10_result[i][0] < value:
                    one_result = (ts_top10_result[i][0], ts_top10_result[i][1], file_name)
                    final_result.append(one_result)
        else:
            continue
    time_end_gpu_cal = datetime.now()
    print('*** GPU & CPU计算用时：%s  %s' % ((time_end_gpu_cal - begin_compare_time), test_file))

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
    print('在 %d 个文件中比对 %s 用时: %s\n\n' % (sources_numb, test_file, (end_time - start_time)))


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
              (i + 1, ts_top10_result[i][0] * 100, ts_top10_result[i][1], int(ts_top10_result[i][1] / 300), (ts_top10_result[i][1] % 300) / 5))


def find_most_probability(results, test_len, value):
    tmp_results = []
    top_results = []

    max_result = 0.15
    max_frame = 0
    interval = test_len - 2

    max_flag = 0
    time1 = datetime.now()
    for i in range(len(results)):
        time10 = datetime.now()
        a = results[i]
        if a > max_result:
            max_result = a
            max_frame = i+1
            max_flag = 1
            continue
        if max_result - a < value and i-max_frame > interval:
            b = (a,i+1)
            tmp_results.append(b)
        # time19 = datetime.now()
        # print("first loop --A:  %s" % (time19 - time10))
    # time2 = datetime.now()
    # print("first loop:  %s" % (time2 - time1))
    if max_flag == 1:
        max = (max_result,max_frame)
        top_results.append(max)
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                b = (tmp_results[i][0],tmp_results[i][1])
                top_results.append(b)
        return(top_results)
    else:
        return (None)


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
        init_val +=  abs(3 - abs(src_frame_feature[1] - test_frame_feature[1]) * 5)
    if src_frame_feature[0] == test_frame_feature[2]:
        init_val += 1 - abs(src_frame_feature[1] - test_frame_feature[3])
    if src_frame_feature[2] == test_frame_feature[0]:
        init_val += 1 - abs(src_frame_feature[3] - test_frame_feature[1])
    if src_frame_feature[2] == test_frame_feature[2]:
        init_val += 1 - abs(src_frame_feature[3] - test_frame_feature[3])
    return init_val


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


def read_sources_features_from_dir(features_dir, fun_features_str2tuple):
    time_begin = datetime.now()
    files = get_files_from_dir(features_dir)
    file_feature_dict = {}
    for file in files:
        if not file.endswith('.txt'):
            continue
        if os.path.isdir('%s/%s' % (features_dir, file)) or file.startswith('.') or os.path.getsize('%s/%s' % (features_dir, file)) == 0:
            print('忽略空文件: %s' % file)
            continue

        source_file = '%s/%s' % (features_dir, file)
        print('src-file %s' % source_file)
        list_of_features = text_read(source_file)
        source_features = np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float)
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
    sources_dir = '/home/para/data/source'  # 比对 所有源的特征值文本 存放路径
    # result_dir = '/storage/auto_test/result_log/'  # 比对log存放路径
    d_sources_features_on_gpu, file_feature_dict = load_features_to_gpu(sources_dir)
    test_dir = '/home/para/data/test'
    test_files = get_files_from_dir(test_dir)

    for test_file in test_files:
        compare('./data/test/%s' % test_file, d_sources_features_on_gpu, file_feature_dict, value)

