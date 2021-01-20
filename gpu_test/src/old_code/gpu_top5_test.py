import datetime

from numba import cuda
import numpy as np
import math


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def to_tuple(data):
    ds = data.split()
    return float(ds[0]), float(ds[1]), float(ds[2]), float(ds[3])


def test_data_to_tuple(data):
    ds = data.split()
    return float(ds[1]), float(ds[2]), float(ds[3]), float(ds[4])


def search_file(test, source):
    time_start_load_data = datetime.datetime.now()

    list_source = text_read(source)
    list_test = text_read(test)

    time_end_load_data = datetime.datetime.now()
    print('读文件用时：%s' % (time_end_load_data - time_start_load_data))

    source_features = np.array(list(map(lambda d: to_tuple(d), list_source)))
    test_features = np.array(list(map(lambda d: test_data_to_tuple(d), list_test)))

    source_features_len = len(source_features)
    test_features_len = len(test_features)

    source_len = source_features_len - test_features_len
    test_len = test_features_len
    total_cal_numb = source_len * test_len
    print('source_features_len=%d source_len=%d test_len=%d  total_cal_numb=%d'
          % (source_features_len, source_len, test_len, total_cal_numb))

    results = np.array([0.0] * source_len, dtype=np.float)
    threads_per_block = 1024    # TODO: 应从设备信息获取
    blocks_per_grid = math.ceil(total_cal_numb / threads_per_block)
    total_thread_numb = threads_per_block * blocks_per_grid
    print('threads_per_block=%d  blocks_per_grid=%d total_thread_numb=%d'
          % (threads_per_block, blocks_per_grid, total_thread_numb))

    if total_cal_numb > total_thread_numb:
        print('Error! 需要调整算法支持 total_cal_numb=%d  total_thread_numb=%d ！！！' % (total_cal_numb, total_thread_numb))

    time_start_h2d = datetime.datetime.now()
    # memory copy from Host to Device
    d_source_features = cuda.to_device(source_features)
    d_test_features = cuda.to_device(test_features)
    d_test_features_results = cuda.device_array(test_len)
    d_gpu_result = cuda.device_array(source_len)
    cuda.synchronize()

    time_end_h2d = datetime.datetime.now()
    print('内存copy用时：%s' % (time_end_h2d - time_start_h2d))

    compare_frame_by_kernel[blocks_per_grid, threads_per_block](total_cal_numb,
                                                                test_len,
                                                                d_source_features, d_test_features,
                                                                d_test_features_results,
                                                                d_gpu_result)
    cuda.synchronize()

    time_end_gpu_calculation = datetime.datetime.now()
    print('*** GPU计算用时：%s' % (time_end_gpu_calculation - time_end_h2d))

    d_gpu_result.copy_to_host(results)
    cuda.synchronize()

    result = 0
    result_frame = 0
    for i in range(source_features_len - test_len):
        a[i] = results[i] / m / 8
        if results[i] > result:
            result_frame = i + 1
            result = results[i]

    time_end_cal = datetime.datetime.now()
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))
    print('*** 匹配用时：%s' % (time_end_cal - time_end_h2d))
    # assert(result == 22.360765)
    # assert(result_frame == 4029)

    return result, result_frame


@cuda.jit
def compare_frame_by_kernel(total_cal_numb, test_len,
                            d_src_frames_features, d_test_frames_features,
                            d_test_features_results, d_results):
    """
     GPU kernel函数。
    :param total_cal_numb:
    :param test_len:
    :param d_src_frames_features: 源视频特征数据序列。Array of tuple
    :param d_test_frames_features: 待检测视频的特征数据序列。
    :param d_test_features_results: 待检测视频的特征数据序列。
    :param d_results:
    :return:
    """
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= total_cal_numb:
        return

    src_frame_start_index = idx // test_len
    test_frame_index = idx % test_len

    source_frame_feature = d_src_frames_features[src_frame_start_index + test_frame_index]
    test_frame_feature = d_test_frames_features[test_frame_index]

    frame_compare_result = compare_two_frame_feature(source_frame_feature, test_frame_feature)
    cuda.atomic.add(d_test_features_results, test_frame_index, frame_compare_result)
    cuda.syncthreads()


    cuda.atomic.add(d_results, src_frame_start_index, frame_compare_result)
    print(idx, src_frame_start_index, test_frame_index, d_results[src_frame_start_index])


@cuda.jit(device=True)
def compare_two_frame_feature(src_frame_feature, test_frame_feature):
    """
    计算两帧特征的相似度。
    :param src_frame_feature: 原帧的特征数据。(Top features with possibility. Tuple of float.)
    :param test_frame_feature: 待检测帧的特征数据。
    :return: 比较结果
    """
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

    start_time = datetime.datetime.now()
    source_files = ['../data/冰上恋人_01.txt']
    for src_file in source_files:
        search_file('../data/test008_AIresult_top5.txt', src_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
