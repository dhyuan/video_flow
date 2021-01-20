import time
from datetime import datetime
import sys
from numba import cuda, jit
import numpy as np
import math
import os


def compare(test_file, d_sources_features_on_gpu, file_feature_dict, top_numb):
    start_time = datetime.now()

    sources_numb = len(d_sources_features_on_gpu)
    end_load_source_time = datetime.now()

    test_features = read_test_features_from_file(test_file, trans_test_data_to_tuple)
    d_test_features = load_test_features_to_gpu(test_features)

    test_len = len(test_features)
    threads_per_block = 1024

    begin_compare_time = datetime.now()
    final_result = []
    for file_name, d_source_features in d_sources_features_on_gpu.items():
        #if '熊仔之雄心壮志52_AIresult_top5.txt' != file_name and '熊仔之雄心壮志51_AIresult_top5.txt' != file_name :
        #    continue
        print('%s -- %s' % (test_file, file_name))
        source_features_len = len(d_source_features)
        source_len = source_features_len - test_len

        blocks_per_grid = math.ceil(source_len / threads_per_block)
        print('threads_per_block=%d  blocks_per_grid=%d' % (threads_per_block, blocks_per_grid))

        init_results = np.array([0.0] * source_len, dtype=np.float)
        d_gpu_result = cuda.to_device(init_results)
        #d_gpu_result = cuda.device_array(source_len)
        cuda.synchronize()

        last_idx = source_features_len - 1
        print('%f, %f, %f, %f' % (d_source_features[0][0], d_source_features[0][1], d_source_features[0][2], d_source_features[0][3]))
        print('%f, %f, %f, %f' % (d_source_features[last_idx][0], d_source_features[last_idx][1], d_source_features[last_idx][2], d_source_features[last_idx][3]))
        time_begin_gpu_cal = datetime.now()
        print('--source_features_len=%d source_len=%d  test_len=%d' % (source_features_len, source_len, test_len))
        compare_frame_by_kernel[blocks_per_grid, threads_per_block](source_len, test_len,
                                                                    d_source_features, d_test_features,
                                                                    d_gpu_result)
        h_results = np.array([0.0] * source_len, dtype=np.float)
        d_gpu_result.copy_to_host(h_results)
        cuda.synchronize()
        time_end_gpu_cal = datetime.now()
        print('*** GPU计算用时：%s' % (time_end_gpu_cal - time_begin_gpu_cal))

        #log_data(file_name, file_feature_dict[file_name], h_results, source_len, source_features_len)

        ts_top10_result = find_most_probability_top_n_from_full_result(h_results, test_len, top_numb)
        print_top10(ts_top10_result, top_numb)

        for i in range(top_numb):
            one_result = (ts_top10_result[i][0], ts_top10_result[i][1], file_name)
            final_result.append(one_result)
        final_result.sort(key=take_first, reverse=True)
        final_result = final_result[0:top_numb:1]

        # for i in range(top_numb):
        #     one_result = (ts_top10_result[0][i], ts_top10_result[1][i], file_name)
        #     final_result.append(one_result)
        # final_result.sort(key=take_first, reverse=True)
        # final_result = final_result[0:top_numb:1]

    print_final_top10(final_result, test_file, top_numb)

    end_time = datetime.now()
    print('\n在 %d 个文件中比对 %s 用时: %s' % (sources_numb, test_file, (end_time - begin_compare_time)))
    print('加载原特征库用时: %s' % (end_load_source_time - start_time))
    print("\nTotalTime: %s" % (end_time - start_time))


def log_data(file_name, src_features, h_results, source_len, source_features_len):
    print('RL file: %s  source_len=%d' % (file_name, source_len))
    for i in range(source_len):
        print('\n%d  %f, %f, %f, %f' % (i, src_features[i][0], src_features[i][1], src_features[i][2], src_features[i][3]))
        print('%f' % (h_results[i]))


def print_final_top10(top10_result, file_name, top_numb):
    print('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：' % file_name)
    f = open(result_dir + '\\GPU_search_result.txt', 'a+')
    f.write('\n\n-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：\n' % file_name)
    for i in range(top_numb):
        print('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' % (
            i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
            (top10_result[i][1] % 300) / 5))
        f.write('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处\n' % (
            i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
            (top10_result[i][1] % 300) / 5))
    f.close()


def take_first(elem):
    return elem[0]


def print_top10(ts_top10_result, top_numb):
    print('AAA top_numb=%d' % top_numb)
    print(ts_top10_result)
    for i in range(top_numb):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' %
              (i + 1, ts_top10_result[i][0] * 100, ts_top10_result[i][1], int(ts_top10_result[i][1] / 300), (ts_top10_result[i][1] % 300) / 5))


def find_most_probability_top_n_from_result(results, start_index, end_index, test_len, top_numb):
    """
    从result全集中，找到top N。根据test_len长度的一半去重。

    :param results: source feature s和 test features 依次比较的全部结果。
    :param start_index: result开始计算的index。
    :param end_index: result停止计算的index。
    :param test_len: 待检测视频的长度（帧数）。
    :param top_numb: 取前 top N 的结果。
    :return:
    """
    print('top_numb=%d' % top_numb)

    top_results = [(0.0, 0)] * top_numb
    frame_interval = int(test_len/2)
    for result_index in range(start_index, end_index):
        a = results[result_index]
        if a < top_results[-1][0]:
            continue
        else:
            same_flag = 0
            for num in range(top_numb):
                if result_index + 1 - top_results[num][1] < frame_interval:
                    same_flag = 1
                    if top_results[num][0] < a:
                        top_results[num] = (a, result_index + 1)
                        top_results.sort(key=take_first, reverse=True)
                        break
                    else:
                        break
            if same_flag == 0:
                top_results[-1] = (a, result_index + 1)
                top_results.sort(key=take_first, reverse=True)
    print(top_results)
    return top_results


def find_most_probability_top_n_from_full_result(results, test_len, top_numb):
    return find_most_probability_top_n_from_result(results, 0, len(results), test_len, top_numb)


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
        if file.startswith('.') or file == '1':
            continue

        source_file = '%s/%s' % (features_dir, file)
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
    lines = open(f, 'r').readlines()
    return lines


def trans_source_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def trans_test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def is_valid_file(file):
    base_name = os.path.basename(file)
    if base_name.startswith('.'):
        return False
    return os.path.isfile(file) and os.path.getsize(file) > 0


def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    valid_files = list(filter(lambda src_file: is_valid_file('%s/%s' % (src_dir, src_file)), files))
    return valid_files


if __name__ == '__main__':
    sources_dir = './data/source'
    test_dir = './data/test'
    result_dir = './result-log'
    test_files = get_files_from_dir(test_dir)

    d_sources_features_on_gpu, file_feature_dict = load_features_to_gpu(sources_dir)

    time_start = time.time()
    for test_file in test_files:
        compare('%s/%s' % (test_dir, test_file), d_sources_features_on_gpu, file_feature_dict, 10)

    time_end = time.time()
    print('所有test比对共耗时：%.2f' % (time_end - time_start))
    f = open(result_dir + '\\GPU_search_result.txt', 'a+')
    f.write('\n所有test比对共耗时：%.2f' % (time_end - time_start))
    f.close()


    #compare('./data/test/test0001_AIresult_top5.txt', d_sources_features_on_gpu, file_feature_dict, 10)





