import datetime

import numpy as np
from numba import cuda, jit


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def to_tuple(data):
    ds = data.split()
    return (int(ds[0]), int(ds[2])), (float(ds[1]), float(ds[3]))


def search_file(test, source):
    start_read_time = datetime.datetime.now()
    list_source = text_read(source)
    list_test = text_read(test)
    end_read_time = datetime.datetime.now()

    source_obj_data = np.array(list(map(lambda d: to_tuple(d)[0], list_source)))
    source_prob_data = np.array(list(map(lambda d: to_tuple(d)[1], list_source)), dtype=np.float)
    test_obj_data = np.array(list(map(lambda d: to_tuple(d)[0], list_test)))
    test_prob_data = np.array(list(map(lambda d: to_tuple(d)[1], list_test)), dtype=np.float)

    # list_src_data = list(map(lambda d: to_tuple(d), list_source))
    # list_test_data = list(map(lambda d: to_tuple(d), list_test))

    print('读文件用时：%s' % (end_read_time - start_read_time))

    n = len(source_obj_data)
    m = len(test_obj_data)
    print('n=%d  m=%d' % (n, m))
    a = np.array([0.0] * (n - m), dtype=np.float)
    result = 0
    result_frame = 0
    print('\n\n')
    beg_cal_without_jit = datetime.datetime.now()
    for i in range(n - m):
        for j in range(m):
            core_cal(a, i, j, source_obj_data, source_prob_data, test_obj_data, test_prob_data)
        if a[i] > result:
            result_frame = i + 1
            result = a[i]
    end_time_without_jit = datetime.datetime.now()
    print('不使用 @jit 匹配用时：%s' % (end_time_without_jit - beg_cal_without_jit))
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))

    print('\n')
    beg_cal_with_jit = datetime.datetime.now()
    for i in range(n - m):
        for j in range(m):
            core_cal_with_jit(a, i, j, source_obj_data, source_prob_data, test_obj_data, test_prob_data)
        if a[i] > result:
            result_frame = i + 1
            result = a[i]
    end_time_with_jit = datetime.datetime.now()
    print('使用 @jit 匹配用时：%s' % (end_time_with_jit - beg_cal_with_jit))
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))


    return result, result_frame


def core_cal(a, i, j, src_obj_id, src_obj_prob, test_obj_id, test_obj_prob):
    s_id = src_obj_id[i + j]
    t_id = test_obj_id[j]
    src_obj_probs = src_obj_prob[i + j]
    test_obj_probs = test_obj_prob[j]

    if s_id[0] == t_id[0]:
        a[i] = a[i] + (src_obj_probs[0] + test_obj_probs[0])
    elif s_id[0] == t_id[1]:
        a[i] = a[i] + (src_obj_probs[0] + test_obj_probs[1])
    elif s_id[1] == t_id[0]:
        a[i] = a[i] + (src_obj_probs[1] + test_obj_probs[0])
    elif s_id[1] == t_id[1]:
        a[i] = a[i] + (src_obj_probs[1] + test_obj_probs[1])


@jit
def core_cal_with_jit(a, i, j, src_obj_id, src_obj_prob, test_obj_id, test_obj_prob):
    s_id = src_obj_id[i + j]
    t_id = test_obj_id[j]
    src_obj_probs = src_obj_prob[i + j]
    test_obj_probs = test_obj_prob[j]

    if s_id[0] == t_id[0]:
        a[i] = a[i] + (src_obj_probs[0] + test_obj_probs[0])
    elif s_id[0] == t_id[1]:
        a[i] = a[i] + (src_obj_probs[0] + test_obj_probs[1])
    elif s_id[1] == t_id[0]:
        a[i] = a[i] + (src_obj_probs[1] + test_obj_probs[0])
    elif s_id[1] == t_id[1]:
        a[i] = a[i] + (src_obj_probs[1] + test_obj_probs[1])


def print_gpu_info():
    print(cuda.gpus)


if __name__ == '__main__':
    print_gpu_info()

    start_time = datetime.datetime.now()
    # source_files = ['./data/一公升的眼泪_log.txt', './data/唐顿庄园_log.txt']
    source_files = ['./data/一公升的眼泪_log.txt']
    for src_file in source_files:
        search_file('./data/test1_log.txt', src_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
