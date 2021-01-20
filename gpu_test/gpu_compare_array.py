import time
from datetime import datetime
import sys
from numba import cuda, float32, int32
import numpy as np
import math
import os

BLOCK_SIZE = 512
BLOCK_TOP_N = 2

BLOCK_RESULTS_SIZE = BLOCK_SIZE * BLOCK_TOP_N
BLOCK_RESULTS_REC_SIZE = 2 * 1

COMPARE_THRESHOLD = 0.05


def compare(test_file_name, d_test_features, d_sources_features, file_feature_dict, threshold):
    sources_numb = len(d_sources_features)
    test_len = len(d_test_features)
    threads_per_block = BLOCK_SIZE

    begin_compare_time = datetime.now()
    final_top_n_result = []
    final_result = []
    max_result = 0.0
    max_flag = 0
    for src_file_name, d_source_features in d_sources_features.items():
        source_features_len = len(d_source_features)
        source_len = source_features_len - test_len

        blocks_numb = math.ceil(source_len / threads_per_block)
        print('src_len=%d test_len=%d threads_per_block=%d  blocks_numb=%d block_top_n=%d'
              % (source_len, test_len, threads_per_block, blocks_numb, BLOCK_TOP_N))

        # 每个 block 保存前 top_numb 的数据。  blocks_numb*top_numb 行， 2 列
        results_on_all_block = np.zeros((blocks_numb * BLOCK_TOP_N, 2), dtype=np.float32)
        d_results_on_all_blocks = cuda.to_device(results_on_all_block)

        cuda.synchronize()

        # 打印源feature的头尾，仅用于debug
        last_idx = source_features_len - 1
        # print('%f, %f, %f, %f' % (d_source_features[0][0], d_source_features[0][1], d_source_features[0][2], d_source_features[0][3]))
        # print('%f, %f, %f, %f' % (d_source_features[last_idx][0], d_source_features[last_idx][1], d_source_features[last_idx][2], d_source_features[last_idx][3]))

        time_begin_gpu_cal = datetime.now()
        # print('--source_features_len=%d source_len=%d  test_len=%d' % (source_features_len, source_len, test_len))

        # 用GPU比对source 和 test，结果保存到 d_results_on_all_blocks
        compare_frame_by_kernel[blocks_numb, threads_per_block](source_len, test_len,
                                                                d_source_features, d_test_features,
                                                                d_results_on_all_blocks, BLOCK_TOP_N)

        results_on_all_block = np.zeros((blocks_numb * BLOCK_TOP_N, 2), dtype=np.float32)
        d_results_on_all_blocks.copy_to_host(results_on_all_block)
        cuda.synchronize()

        time_end_gpu_cal = datetime.now()
        print('*** GPU计算用时：%s  %s --> %s' % ((time_end_gpu_cal - time_begin_gpu_cal), test_file, src_file_name))

        # 从 results_on_all_block 中找到 topN。
        if test_file.endswith('Aest0001_AIresult_top5.txt') and (src_file_name.endswith('9秒危机_AIresult_top5.txt')
                or src_file_name.endswith('8毫米_AIresult_top5.txt')):
            print("====")
            print(src_file_name)
            print(results_on_all_block)
        ts_top_n_result = find_most_probability_with_threshold(results_on_all_block, test_len, threshold)
        if test_file.endswith('Aest0001_AIresult_top5.txt') and src_file_name.endswith('9秒危机_AIresult_top5.txt'):
            print_top10(ts_top_n_result, 5)

        if ts_top_n_result != None:
            if ts_top_n_result[0][0] > max_result:
                max_result = ts_top_n_result[0][0]
                max_flag = 1
            for i in range(len(ts_top_n_result)):
                if max_result - ts_top_n_result[i][0] < threshold:
                    one_result = (ts_top_n_result[i][0], ts_top_n_result[i][1], src_file_name)
                    final_result.append(one_result)
        else:
            continue

    if max_flag == 1:
        output_final_result = []
        for i in range(len(final_result)):
            if max_result - final_result[i][0] < threshold:
                one_result = (final_result[i][0], final_result[i][1], final_result[i][2])
                output_final_result.append(one_result)

        output_final_result.sort(key=take_first, reverse=True)
        print_final_top10(output_final_result, test_file)
        print('----------最大概率为：%.2f%%--------------------'%(max_result*100))
    else:
        print_final_top10(None, test_file)
        print('---------------最大概率小于15%，结果为空-------------------------------')

    end_compare_time = datetime.now()
    print('在 %d 个文件中比对 %s 用时: %s \n' % (sources_numb, test_file_name, (end_compare_time - begin_compare_time)))


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


def find_most_probability_with_threshold(results, test_len, value):
    tmp_results = []
    top_results = []
    max_result = 0.15
    max_frame = 0
    interval = test_len - 2
    max_flag = 0
    for i in range(len(results)):
        a = results[i][0]
        if a > max_result:
            max_result = a
            max_frame = results[i][1] + 1
            max_flag = 1
            continue
        if max_result - a < value and results[i][1] + 1 - max_frame > interval:
            b = (a, results[i][1] + 1)
            tmp_results.append(b)
    if max_flag == 1:
        max = (max_result, max_frame)
        top_results.append(max)
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                b = (tmp_results[i][0],tmp_results[i][1])
                top_results.append(b)
        return (top_results)
    else:
        return (None)


def log_data(file_name, src_features, h_results, source_len, source_features_len):
    print('RL file: %s  source_len=%d' % (file_name, source_len))
    for i in range(source_len):
        print('\n%d  %f, %f, %f, %f' % (i, src_features[i][0], src_features[i][1], src_features[i][2], src_features[i][3]))
        print('%f' % (h_results[i]))


def take_first(elem):
    return elem[0]


def print_top10(frame_probabilities, top_numb):
    numb = min(top_numb, len(frame_probabilities))
    print('---> %d  %d ' % (top_numb, len(frame_probabilities)))
    for i in range(numb):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' %
              (i + 1, frame_probabilities[i][0] * 100, frame_probabilities[i][1], int(frame_probabilities[i][1] / 300), (frame_probabilities[i][1] % 300) / 5))



@cuda.jit
def compare_frame_by_kernel(source_len, test_len,
                            d_src_frames_features, d_test_frames_features,
                            d_all_block_results, block_top_k):
    # 在share memory中构建一个二维数组保存这个线程块的 topK。TODO：为什么每个core都执行这个？ numba如何保证块内唯一？
    s_block_result = cuda.shared.array((BLOCK_RESULTS_SIZE, BLOCK_RESULTS_REC_SIZE), dtype=float32)
    s_block_result[cuda.threadIdx.x][0] = 0
    s_block_result[cuda.threadIdx.x][1] = 0

    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= source_len:
        return

    # 线程idx就是当前线程比较： 源的位置 <--> d_test_frames_features。
    src_frame_start_index = idx
    result_at_idx = calculate_features_probability_at_src_idx(src_frame_start_index, test_len,
                                                              d_src_frames_features, d_test_frames_features)
    s_block_result[cuda.threadIdx.x][0] = result_at_idx
    s_block_result[cuda.threadIdx.x][1] = idx

    # 等待同一 block 的所有线程都计算完毕。
    cuda.syncthreads()

    # 线程块的第一个线程负责找出块内 topK.  TODO：因为cuda.jit方法传递的参数类型不能用于 cuda.local.array 创建数组时用于指定维数，所以这里使用全集变量。
    if cuda.threadIdx.x == 0:
        set_top_n_possible_of_block_to_results(s_block_result, d_all_block_results)


@cuda.jit(device=True)
def set_top_n_possible_of_block_to_results(s_block_result, d_results):
    """
    在 s_block_result 保存的每个线程的计算结果找出top_n，放到 d_results 中。

    :param s_block_result:
    :param d_results:
    :return:
    """

    valid_result_size = cuda.local.array(1, dtype=float32)
    top_n_in_block = gpu_find_most_probability_with_threshold(s_block_result, BLOCK_TOP_N, float32(0.05), valid_result_size)
    # top_n_in_block = s_block_result[0:BLOCK_TOP_N]
    for i in range(BLOCK_TOP_N):
        cuda.atomic.add(d_results[cuda.blockIdx.x * BLOCK_TOP_N + i], 0, top_n_in_block[i][0])
        cuda.atomic.add(d_results[cuda.blockIdx.x * BLOCK_TOP_N + i], 1, top_n_in_block[i][1])

    # // 只取最大值
    # block_max = (s_block_result[0][0], s_block_result[0][1])
    # for i in range(1, BLOCK_SIZE):
    #     if block_max[0] < s_block_result[i][0]:
    #         block_max = (s_block_result[i][0], s_block_result[0][1])
    # cuda.atomic.add(d_results[cuda.blockIdx.x], 0, block_max[0])
    # cuda.atomic.add(d_results[cuda.blockIdx.x], 1, block_max[1])


@cuda.jit(device=True)
def gpu_find_most_probability_with_threshold(results, test_len, value):
    tmp_results = []
    top_results = []
    max_result = 0.15
    max_frame = 0
    interval = test_len - 2
    max_flag = 0

    for i in range(len(results)):
        frame_value = results[i][0]
        frame_index = results[i][1] + 1
        if frame_value > max_result:
            max_result = frame_value
            max_frame = frame_index
            max_flag = 1
            continue
        if max_result - frame_value < value and i - max_frame > interval:
            b = (frame_value, frame_index)
            tmp_results.append(b)

    if max_flag == 1:
        max = (max_result, max_frame)
        top_results.append(max)
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                b = (tmp_results[i][0], tmp_results[i][1])
                top_results.append(b)
        return (top_results)
    else:
        return (None)

# @cuda.jit(device=True)
# def find_top_n(result_data, top_k):
#     for i in range(1, top_k):
#         for j in range(i, 0, -1):
#             if result_data[j][0] > result_data[j - 1][0]:
#                 result_data[j][0], result_data[j - 1][0] = result_data[j - 1][0], result_data[j][0]
#                 result_data[j][1], result_data[j - 1][1] = result_data[j - 1][1], result_data[j][1]
#             else:
#                 pass
#     for i in range(top_k, len(result_data)):
#         if result_data[i][0] > result_data[top_k - 1][0]:
#             result_data[top_k - 1][0] = result_data[i][0]
#             result_data[top_k - 1][1] = result_data[i][1]
#             for j in range(top_k - 1, 0, -1):
#                 if result_data[j][0] > result_data[j - 1][0]:
#                     result_data[j][0], result_data[j - 1][0] = result_data[j - 1][0], result_data[j][0]
#                     result_data[j][1], result_data[j - 1][1] = result_data[j - 1][1], result_data[j][1]
#                 else:
#                     pass
#     # 这个cuda.jit方法是不能返回下面这样的值
#     # return result_data[0:top_k]


@cuda.jit(device=True)
def calculate_features_probability_at_src_idx(source_start_index, test_len,
                                              d_src_frames_features, d_test_frames_features):
    """
    从源d_src_frames_features的指定位置 source_start_index 跟 d_test_frames_features 进行比较。

    :param source_start_index:
    :param test_len:
    :param d_src_frames_features:
    :param d_test_frames_features:
    :return:
    """
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


def is_valid_source_file(file_name):
    base_name = os.path.basename(file_name)
    if os.path.isdir(file_name) or base_name.startswith('.') or os.path.getsize(file_name) == 0:
        return False
    return True


def read_sources_features_from_dir(features_dir, fun_features_str2tuple):
    time_begin = datetime.now()
    files = get_files_from_dir(features_dir)
    source_file_feature_dict = {}
    for file in files:
        if not is_valid_source_file('%s/%s' % (features_dir, file)):
            continue

        source_file = '%s/%s' % (features_dir, file)
        list_of_features = text_read(source_file)
        source_features = np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float)
        source_file_feature_dict[file] = source_features
    time_end = datetime.now()
    print('读取%d个源特征文件用时：%s' % (len(files), (time_end - time_begin)))
    return source_file_feature_dict


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
    test_files = get_files_from_dir(test_dir)

    start_load_data_time = datetime.now()
    d_sources_features_on_gpu, file_feature_dict = load_features_to_gpu(sources_dir)

    start_time = datetime.now()
    for test_file in test_files:
        test_features = read_test_features_from_file('%s/%s' % (test_dir, test_file), trans_test_data_to_tuple)
        d_test_features_on_gpu = load_test_features_to_gpu(test_features)
        compare(test_file, d_test_features_on_gpu, d_sources_features_on_gpu, file_feature_dict, COMPARE_THRESHOLD)

    end_time = datetime.now()
    print('TOTAL_TIME: load_data=%s   compute=%s' % ((start_time - start_load_data_time), (end_time - start_time)))

    #compare('./data/test/test0001_AIresult_top5.txt', d_sources_features_on_gpu, file_feature_dict, 10)


