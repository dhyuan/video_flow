#!/usr/bin/env python
# coding:utf-8
import math
import os
from datetime import datetime

import numpy as np
from numba import cuda, float32, int32, literally


def compare(d_test_features, d_sources_features, files_features_index_dict, file_index_dict, threads_per_block=1024):
    """
    使用GPU计算"待检测视频特征"在"特征合集"各位置上的概率。返回的数组长度是： 特征合集长度 - 待测视频特征长度

    :param d_test_features: GPU上的待检测的特征集
    :param d_sources_features: GPU上的"特征合集"
    :param files_features_index_dict: 源特征文件id所包含的特征在特征合集中的起始位置(含)、结束位置(不含)的关系。
    :param threads_per_block: 每个GPU Block 上的线程数。
    :return: 以视频"特征合集"中每帧为起点，对应的概率，返回的数组长度是： 特征合集长度 - 待测视频特征长度
    """
    # 待测特征长度
    test_len = len(d_test_features)

    # 特征合集长度
    total_source_features_len = len(d_sources_features)

    # 需要多少Block
    total_blocks = math.ceil(total_source_features_len / threads_per_block)

    begin_block_index = 0
    begin_index_in_block = 0
    print('\nSourceFilesNumb=%d,  SourceFeatures=%d, TotalBlocks=%d, test_len=%d'
          % (len(files_features_index_dict), total_source_features_len, total_blocks, test_len))

    # 计算每个源文件对应的 起始block以及在该block上对应的thread-index、以及 结尾block以及在该block上对应的thread-index

    file_block_index_array = []
    for file_index in range(len(files_features_index_dict)):
        feature_data = files_features_index_dict[file_index]
        begin_index = feature_data[0]   # 每源文件的起始特征在特征合集中的位置
        end_index = feature_data[1]     # 每源文件的结束特征在特征合集中的位置

        # 此源文件最后特征所在的 block。 end_index - test_len - 1 是最后一个有效帧的位置索引。
        end_block_index = math.floor((end_index - test_len - 1) / threads_per_block)

        # 此源文件最后特征所在的block的 thread index
        end_index_in_block = (end_index - test_len - 1) % threads_per_block

        # 左右闭区间
        file_block_index_array.append([file_index, begin_block_index, begin_index_in_block, end_block_index, end_index_in_block])
        # file_block_thread_map = np.vstack([file_block_thread_map,
        #                                    [file_index, begin_block_index, begin_index_in_block, end_block_index,
        #                                     end_index_in_block]])

        # 下一个源文件起始的block index
        begin_block_index = math.floor(end_index / threads_per_block)

        # 下一个源文件起始的block的 thread index
        begin_index_in_block = end_index % threads_per_block

    time_start_gpu_cal = datetime.now()

    file_block_thread_map = np.array(file_block_index_array, dtype=np.int32)
    total_block_num = len(file_block_thread_map)
    # 把源文件特征在"特征合集"中位置信息传给GPU。
    to_device_time0 = datetime.now()
    d_file_block_thread_map = cuda.to_device(file_block_thread_map)
    cuda.synchronize()
    print('--- Transfer file_block_thread_map to GPU.  BlocksNumb=%d  BlockIndexInfoSize=%d' % (total_block_num, len(d_file_block_thread_map)))

    # 定义保存结果的数组，传给GPU
    # h_results = np.zeros((total_source_features_len, 3), dtype=np.float32)
    h_results = np.zeros((total_block_num, 3), dtype=np.float32)
    d_gpu_result = cuda.to_device(h_results)
    to_device_time1 = datetime.now()
    print('Trans data to GPU time: %s' % (to_device_time1 - to_device_time0))
    compare_frame_by_kernel[total_blocks, threads_per_block](total_source_features_len, test_len,
                                                             d_file_block_thread_map,
                                                             d_sources_features, d_test_features,
                                                             d_gpu_result)
    cuda.synchronize()
    gpu_compute_end_time = datetime.now()
    print('GPU pure compute time: %s' % (gpu_compute_end_time - to_device_time1))
    # 把结果从GPU取回来
    d_gpu_result.copy_to_host(h_results)
    cuda.synchronize()

    # print(h_results)

    time_end_gpu_cal = datetime.now()
    print('Trans data back to CPU time: %s' % (time_end_gpu_cal - gpu_compute_end_time))
    print('*** GPU计算用时：%s' % (time_end_gpu_cal - time_start_gpu_cal))
    return h_results


def debug(h_debug_info):
    print(h_debug_info)
    err_indexes = h_debug_info[np.where(h_debug_info > 0)]
    min_index = np.min(err_indexes)
    print('err_size=%d  minIndex=%d' % (len(err_indexes), min_index))


def calculate_result(test_file, files_index, h_results, test_len, value):
    cpu_find_result_begin_time = datetime.now()

    ts_top10_result = find_most_probability(h_results, test_len, value)

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

    if max_flag == 1:
        output_final_result = []
        for i in range(len(final_result)):
            if max_result - final_result[i][0] < value:
                one_result = (final_result[i][0], final_result[i][1], final_result[i][2])
                output_final_result.append(one_result)

        output_final_result.sort(key=take_first, reverse=True)
        print_final_top10(output_final_result, test_file)
        print('FOUND ----------最大概率为：%.2f%%--------------------' % (max_result * 100))
    else:
        print_final_top10(None, test_file)
        print('NO_FOUND-------------最大概率小于15%，结果为空-------------------------------')
    cpu_find_result_end_time = datetime.now()
    print('*** CPU process result 用时：%s' % (cpu_find_result_end_time - cpu_find_result_begin_time))


@cuda.jit
def compare_frame_by_kernel(total_source_features_len, test_len, d_file_block_thread_map,
                            d_src_frames_features, d_test_frames_features, d_results):

    s_block_result = cuda.shared.array(shape=(1024, 3), dtype=float32)

    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= total_source_features_len:
        return

    file_b_t_map_size = len(d_file_block_thread_map)
    s_file_block_thread_map = cuda.shared.array(shape=(1200, 5), dtype=float32)
    if cuda.threadIdx.x == 0:
        for i in range(1200):
            s_file_block_thread_map[i][0] = d_file_block_thread_map[i][0]
            s_file_block_thread_map[i][1] = d_file_block_thread_map[i][1]
            s_file_block_thread_map[i][2] = d_file_block_thread_map[i][2]
            s_file_block_thread_map[i][3] = d_file_block_thread_map[i][3]
            s_file_block_thread_map[i][4] = d_file_block_thread_map[i][4]
    cuda.syncthreads

    for i in range(file_b_t_map_size):
        index_scope = s_file_block_thread_map[i]
        file_index = index_scope[0]
        begin_block_idx = index_scope[1]
        begin_t_idx_in_block = index_scope[2]
        end_block_idx = index_scope[3]
        end_t_idx_in_block = index_scope[4]

        begin_index = cuda.blockDim.x * begin_block_idx + begin_t_idx_in_block
        end_index = cuda.blockDim.x * end_block_idx + end_t_idx_in_block

        # 匹配当前 thread 的工作。
        if begin_index <= idx <= end_index:
            result_at_idx = calculate_features_probability_at_src_idx(idx, test_len, d_src_frames_features,
                                                                      d_test_frames_features)
            frame_index = idx - cuda.blockDim.x * begin_block_idx - begin_t_idx_in_block
            # cuda.atomic.add(d_results[idx], 0, result_at_idx)
            # cuda.atomic.add(d_results[idx], 1, frame_index)
            # cuda.atomic.add(d_results[idx], 2, file_index)

            s_block_result[cuda.threadIdx.x][0] = result_at_idx
            s_block_result[cuda.threadIdx.x][1] = frame_index
            s_block_result[cuda.threadIdx.x][2] = file_index


            # 汇总最大值, 由 block 的第一个 thread 计算当前block的max
            cuda.syncthreads()
            if cuda.threadIdx.x == 0:
                block_max = s_block_result[0][0]
                block_max_frame_index = s_block_result[0][1]
                block_max_file_index = s_block_result[0][2]
                for k in range(1, cuda.blockDim.x):
                    if block_max < s_block_result[k][0]:
                        block_max = s_block_result[k][0]
                        block_max_frame_index = s_block_result[k][1]
                        block_max_file_index = s_block_result[k][2]
                cuda.atomic.add(d_results[cuda.blockIdx.x], 0, block_max)
                cuda.atomic.add(d_results[cuda.blockIdx.x], 1, block_max_frame_index)
                cuda.atomic.add(d_results[cuda.blockIdx.x], 2, block_max_file_index)
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


def find_most_probability(results, test_len, value):
    tmp_results = np.empty((0, 3), dtype=np.float32)
    top_results = np.empty((0, 3), dtype=np.float32)

    max_result = 0.15
    max_frame = 0
    max_file_index = -1

    interval = test_len - 2
    max_flag = 0

    possible_results = results[np.where((results[:, 0] > max_result))]
    print('len of from %d to %d' % (len(results), len(possible_results)))

    results = possible_results

    for i in range(len(results)):
        frame_probability = results[i][0]
        frame_index = results[i][1] + 1
        file_index = results[i][2]
        if frame_probability > max_result:
            max_result = frame_probability
            max_frame = frame_index
            max_file_index = file_index
            max_flag = 1
            continue
        if max_result - frame_probability < value and results[i][1] - max_frame > interval:
            tmp_results = np.vstack([tmp_results, [frame_probability, frame_index, file_index]])

    if max_flag == 1:
        max = (max_result, max_frame, max_file_index)
        top_results = np.vstack([top_results, [max_result, max_frame, max_file_index]])
        for i in range(len(tmp_results)):
            if max_result - tmp_results[i][0] < value:
                top_results = np.vstack([top_results, [tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]]])
                # top_results.append((tmp_results[i][0], tmp_results[i][1], tmp_results[i][2]))
        return (top_results)
    else:
        return (None)


def trans_all_source_features_as_whole(file_features_dict):
    """
    根据输入的'每个文件及对应特征值的字典'，把所有特征值合并在一个大数组中——"特征合集"，并构建以下字典：
    1）files_index_dict：给每个文件一个依次增长的ID值，构建出文件ID及文件名的字典。
    2）files_features_index_dict：以文件ID为key，以这个文件特征值在'特征合集'中的起始位置、结束位置为tuple元素作为value。

    :param file_features_dict: 每个文件及对应特征值的字典。
    :return: 返回 "特征合集"、files_index_dict、files_features_index_dict
    """
    time_begin = datetime.now()

    total_features_numb = 0
    # files_index_dict = {}
    files_features_index_dict = {}
    begin_index = 0
    file_id = 0
    print('prepare trans features')
    for file_name, features in file_features_dict.items():
        source_features_numb = len(features)
        total_features_numb += source_features_numb

        # files_index_dict[file_id] = file_name
        files_features_index_dict[file_id] = (begin_index, total_features_numb)

        begin_index = total_features_numb
        file_id += 1
    time_prepare_end = datetime.now()

    mem_size = total_features_numb * 4 * 4  # 点精度浮点32位，4字节，每个feature 16个字节。
    print('prepare trans features end. TotalFeatures: %d %d %s' % (
        total_features_numb, mem_size, (time_prepare_end - time_begin)))

    # all_features = np.empty((0, 4), dtype=np.float32)
    # for file_id in range(len(files_features_index_dict.items())):
    #     file_name = files_index_dict[file_id]
    #     features = file_features_dict[file_name]
    #     all_features = np.vstack((all_features, features))

    time_prepare_end2 = datetime.now()
    print('prepare all_features end. %s' % (time_prepare_end2 - time_prepare_end))

    return files_features_index_dict


def read_sources_features_from_dir(features_dir, f_features_transformer):
    """
    从保存源特征的目录读取特征值。
    以文件名为key，以特征列表为value的字典返回。每个value是一个tuple，包含取值最高的两个特征及概率。

    :param features_dir:
    :param f_features_transformer:
    :return:
    """
    time_begin = datetime.now()
    files = get_files_from_dir(features_dir)
    file_feature_dict = {}
    counter = 0
    all_features_data_array = []
    file_id_name_dict = {}

    for file in files:
        # if not file.endswith('U2RoseBowl体育演唱会_AIresult_top5.txt') and not file.endswith('2001太空漫游_AIresult_top5.txt'):
        #     continue

        if os.path.isdir('%s/%s' % (features_dir, file)) or not file.endswith('.txt') or file.startswith(
                '.') or os.path.getsize(
                '%s/%s' % (features_dir, file)) == 0:
            print('忽略空文件: %s' % file)
            continue

        source_file = '%s/%s' % (features_dir, file)
        list_of_features = text_read(source_file)
        print('src-file %d  %s %d' % (counter, source_file, len(list_of_features)))

        list_of_features = list(map(lambda d: f_features_transformer(d), list_of_features))
        source_features = np.array(list_of_features, dtype=np.float32)
        file_feature_dict[file] = source_features
        file_id_name_dict[counter] = file
        all_features_data_array.extend(list_of_features)
        # all_features = np.vstack((all_features, features))
        counter += 1

    all_features = np.asarray(all_features_data_array, dtype=np.float32)

    time_end = datetime.now()
    print('读取%d个源特征文件用时：%s' % (len(files), (time_end - time_begin)))
    return file_feature_dict, file_id_name_dict, all_features


def read_test_features_from_file(testfile, fun_features_str2tuple):
    list_of_features = text_read(testfile)
    _test_features = np.array(list(map(lambda d: fun_features_str2tuple(d), list_of_features)), dtype=np.float)
    return _test_features


def copy_features_array_to_gpu(features):
    """
    把numpy数组的内容transfer到GPU，返回GPU上对应数组的引用。

    :param features: 要传输的封装在numpy中的feature数据
    :return: 对GPU上数据的引用
    """
    d_features = cuda.to_device(features)
    return d_features


def text_read(f):
    try:
        lines = open(f, 'r').readlines()
        return lines
    except Exception as e:
        print(e)
        print('ERROR, 结果文件不存在！')


def trans_source_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def trans_test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def trans_source_data_to_list(data):
    ds = data.split()
    return [float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])]


def trans_test_data_to_list(data):
    ds = data.split()
    return [float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])]


def get_files_from_dir(src_dir):
    files = os.listdir(src_dir)
    return files


def demo_compare(test_dir, sources_dir, value=0.05):
    # 读从source features目录读取特征值。
    source_files_features_dict, file_index_dict, all_source_features = read_sources_features_from_dir(sources_dir, trans_source_data_to_tuple)

    # 构建source features的"特征合集"及相关字典。
    file_features_index_dict = trans_all_source_features_as_whole(source_files_features_dict)

    # 传输"特征合集"到GPU。
    d_all_sources_features_in_gpu = copy_features_array_to_gpu(all_source_features)

    test_files = get_files_from_dir(test_dir)
    for test_file in test_files:
        # if not test_file.endswith('test0017_AIresult_top5.txt'):
        #     continue
        compare_start_time = datetime.now()

        # 读取测试文件构建特征值数组
        test_features = read_test_features_from_file('%s/%s' % (test_dir, test_file), trans_test_data_to_tuple)

        # 传输待检测特征到GPU。
        d_test_features_in_gpu = copy_features_array_to_gpu(test_features)

        # 用GPU计算各个位置的概率
        frames_results = compare(d_test_features_in_gpu, d_all_sources_features_in_gpu, file_features_index_dict, file_index_dict)

        # 找出可能性最高的位置
        calculate_result(test_file, file_index_dict, frames_results, len(test_features), value)

        compare_end_time = datetime.now()
        print('在 %d 个文件中比对 %s 用时: %s\n' % (len(file_index_dict), test_file, (compare_end_time - compare_start_time)))


if __name__ == '__main__':
    source_files_dir = './data/source'  # 比对 所有源的特征值文本 存放路径
    source_files_dir = '/storage/auto_test/source_result'  # 比对 所有源的特征值文本 存放路径
    test_files_dir = './data/test'

    demo_compare(test_files_dir, source_files_dir, 0.05)
