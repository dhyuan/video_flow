import datetime

from numba import cuda, jit
import numpy as np
import math


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def to_tuple(data):
    ds = data.split()
    return (int(ds[0]), int(ds[2])), (float(ds[1]), float(ds[3]))


def test_data_to_tuple(data):
    ds = data.split()
    return (int(ds[1]), int(ds[3])), (float(ds[2]), float(ds[4]))


def search_file(test, source):
    time_start_load_data = datetime.datetime.now()

    list_source = text_read(source)
    list_test = text_read(test)

    time_end_load_data = datetime.datetime.now()
    print('读文件用时：%s' % (time_end_load_data - time_start_load_data))

    source_obj_data = np.array(list(map(lambda d: to_tuple(d)[0], list_source)))
    source_prob_data = np.array(list(map(lambda d: to_tuple(d)[1], list_source)), dtype=np.float)
    test_obj_data = np.array(list(map(lambda d: test_data_to_tuple(d)[0], list_test)))
    test_prob_data = np.array(list(map(lambda d: test_data_to_tuple(d)[1], list_test)), dtype=np.float)

    n = len(source_obj_data)
    m = len(test_obj_data)

    layer1 = n - m
    layer2 = m
    total_cal_numb = layer1 * layer2
    print('n=%d  m=%d n-m=%d  totalLoop=%d' % (n, m, (n - m), total_cal_numb))

    results = np.array([0.0] * (n - m), dtype=np.float)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(total_cal_numb / threads_per_block)
    print('threads_per_block=%d  blocks_per_grid=%d' % (threads_per_block, blocks_per_grid))

    time_start_h2d = datetime.datetime.now()
    d_source_obj_data = cuda.to_device(source_obj_data)
    d_source_prob_data = cuda.to_device(source_prob_data)
    d_test_obj_data = cuda.to_device(test_obj_data)
    d_test_prob_data = cuda.to_device(test_prob_data)
    #d_gpu_result = cuda.to_device(results)
    d_gpu_result = cuda.device_array(n - m)
    time_end_h2d = datetime.datetime.now()
    print('内存copy用时：%s' % (time_end_h2d - time_start_h2d))
    
    cuda.synchronize()
    core_cal[blocks_per_grid, threads_per_block](total_cal_numb, layer1, layer2,
                                                 d_source_obj_data, d_source_prob_data,
                                                 d_test_obj_data, d_test_prob_data,
                                                 d_gpu_result)
    cuda.synchronize()
    time_end_gpu_calculation = datetime.datetime.now()
    print('*** GPU计算用时：%s' % (time_end_gpu_calculation - time_end_h2d))

    d_gpu_result.copy_to_host(results)

    result = 0
    result_frame = 0
    for i in range(n - m):
        if results[i] > result:
            result_frame = i + 1
            result = results[i]

    time_end_cal = datetime.datetime.now()
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))
    print('*** 匹配用时：%s' % (time_end_cal - time_end_h2d))
    #assert(result == 22.360765)
    #assert(result_frame == 4029)

    return result, result_frame


@cuda.jit
def core_cal(total_cal_numb, layer1, layer2,
             d_src_objs, d_src_probs,
             d_test_objs, d_test_probs,
             d_results):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx >= total_cal_numb:
        #print(total_cal_numb, idx)
        return

    i = idx // layer2
    j = idx % layer2

    s_id = d_src_objs[i + j]
    t_id = d_test_objs[j]

    src_obj_probs = d_src_probs[i + j]
    test_obj_probs = d_test_probs[j]

    k = 0
    if s_id[0] == t_id[0]:
        k += 6 - abs((src_obj_probs[0] - test_obj_probs[0])) * 18
        cuda.atomic.add(d_results, i, k)
    elif s_id[0] == t_id[1]:
        k += 2 - abs((src_obj_probs[0] - test_obj_probs[1])) * 6
        cuda.atomic.add(d_results, i, k)
    elif s_id[1] == t_id[0]:
        k += 2 - abs((src_obj_probs[1] - test_obj_probs[0])) * 6
        cuda.atomic.add(d_results, i, k)
    elif s_id[1] == t_id[1]:
        k += 2 - abs((src_obj_probs[1] - test_obj_probs[1])) * 6
        cuda.atomic.add(d_results, i, k)
    #print(idx, i, j, d_results[i])


def print_gpu_info():
    print(cuda.gpus)


if __name__ == '__main__':
    print_gpu_info()

    start_time = datetime.datetime.now()
    source_files = ['./data/冰上恋人_01.txt']
    for src_file in source_files:
        search_file('./data/test008_AIresult_top5.txt', src_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
