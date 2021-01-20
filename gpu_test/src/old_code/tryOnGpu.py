import datetime

from numba import cuda, jit
import numpy as np
import math


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def trans_source_data_to_tuple(data):
    ds = data.split()
    return (int(ds[0]), int(ds[2])), (float(ds[1]), float(ds[3]))


def trans_test_data_to_tuple(data):
    ds = data.split()
    return (int(ds[1]), int(ds[3])), (float(ds[2]), float(ds[4]))


def search_file(test, source):
    time_start_load_data = datetime.datetime.now()

    list_source = text_read(source)
    list_test = text_read(test)

    time_end_load_data = datetime.datetime.now()
    print('读文件用时：%s' % (time_end_load_data - time_start_load_data))

    source_features_ids = np.array(list(map(lambda d: trans_source_data_to_tuple(d)[0], list_source)), dtype=np.int)
    source_features_probabilities = np.array(list(map(lambda d: trans_source_data_to_tuple(d)[1], list_source)), dtype=np.float)
    test_features_ids = np.array(list(map(lambda d: trans_test_data_to_tuple(d)[0], list_test)), dtype=np.int)
    test_features_probabilities = np.array(list(map(lambda d: trans_test_data_to_tuple(d)[1], list_test)), dtype=np.float)

    source_features_len = len(source_features_ids)
    test_features_len = len(test_features_ids)

    source_len = source_features_len - test_features_len
    test_len = test_features_len
    print('source_features_len=%d source_len=%d test_len=%d' % (source_features_len, source_len, test_len))

    threads_per_block = 1024  # TODO: 应从设备信息获取
    blocks_per_grid = math.ceil(source_len / threads_per_block)
    total_thread_numb = threads_per_block * blocks_per_grid
    print('threads_per_block=%d  blocks_per_grid=%d total_thread_numb=%d'
          % (threads_per_block, blocks_per_grid, total_thread_numb))

    results = np.array([0.0] * (source_features_len - test_features_len), dtype=np.float)

    # copy memory from Host to Device
    time_start_h2d = datetime.datetime.now()
    d_source_features_ids = cuda.to_device(source_features_ids)
    d_source_features_probabilities = cuda.to_device(source_features_probabilities)
    d_test_features_ids = cuda.to_device(test_features_ids)
    d_test_feature_probabilities = cuda.to_device(test_features_probabilities)
    d_gpu_result = cuda.device_array(source_len)
    time_end_h2d = datetime.datetime.now()
    cuda.synchronize()
    print('内存copy用时：%s' % (time_end_h2d - time_start_h2d))

    total_cal_numb = source_len * test_len

    compare_frame_by_kernel[blocks_per_grid, threads_per_block](source_len, test_len, )
    # core_cal[blocks_per_grid, threads_per_block](total_cal_numb, source_len, test_len,
    #                                              d_source_features_ids, d_source_features_probabilities,
    #                                              d_test_features_ids, d_test_feature_probabilities,
    #                                              d_gpu_result)
    cuda.synchronize()
    time_end_gpu_calculation = datetime.datetime.now()
    print('*** GPU计算用时：%s' % (time_end_gpu_calculation - time_end_h2d))

    d_gpu_result.copy_to_host(results)

    result = 0
    result_frame = 0
    for i in range(source_features_len - test_features_len):
        if results[i] > result:
            result_frame = i + 1
            result = results[i]

    time_end_cal = datetime.datetime.now()
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))
    print('*** 匹配用时：%s' % (time_end_cal - time_end_h2d))
    #assert(result == 22.360765)
    #assert(result_frame == 4029)

    return result, result_frame


# @cuda.jit
# def core_cal(total_cal_numb, layer1, layer2,
#              d_src_objs, d_src_probs,
#              d_test_objs, d_test_probs,
#              d_results):
#     idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#     if idx >= total_cal_numb:
#         #print(total_cal_numb, idx)
#         return
#
#     i = idx // layer2
#     j = idx % layer2
#
#     s_id = d_src_objs[i + j]
#     t_id = d_test_objs[j]
#
#     source_frame_feature = d_src_probs[i + j]
#     test_frame_feature = d_test_probs[j]
#
#     calculate_features_probability_at_src_idx


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
    for test_idx in np.arange(test_len):
        src_idx = source_start_index + test_idx
        accumulator = compare_two_frame_feature(d_src_frames_features[src_idx], d_test_frames_features[test_idx], accumulator)
    result_of_from_idx = accumulator / test_len / 8
    return result_of_from_idx


@cuda.jit(device=True)
def compare_two_frame_feature(src_frame_feature, test_frame_feature):
    k = 0
    if src_frame_feature[0] == test_frame_feature[0]:
        k += 6 - abs((src_frame_feature[1] - test_frame_feature[1])) * 18
    elif src_frame_feature[0] == test_frame_feature[2]:
        k += 2 - abs((src_frame_feature[1] - test_frame_feature[3])) * 2
    elif src_frame_feature[2] == test_frame_feature[0]:
        k += 2 - abs((src_frame_feature[3] - test_frame_feature[1])) * 2
    elif src_frame_feature[2] == test_frame_feature[2]:
        k += 2 - abs((src_frame_feature[3] - test_frame_feature[3])) * 2
    return k



def print_gpu_info():
    print(cuda.gpus)


if __name__ == '__main__':
    print_gpu_info()

    for i in range(5):
        start_time = datetime.datetime.now()
        source_files = ['./data/冰上恋人_01.txt']

        for src_file in source_files:
            search_file('./data/test008_AIresult_top5.txt', src_file)

        end_time = datetime.datetime.now()
        print("\n\nTotalTime: %s" % (end_time - start_time))
