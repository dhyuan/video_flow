import numpy as np


def find_top_n(array_data, top_k):
    """
    对array_data进行排序，前top_k个元素是最大值。

    :param array_data:
    :param top_k:
    :return:
    """
    for i in range(1, top_k):
        for j in range(i, 0, -1):
            if array_data[j][0] > array_data[j - 1][0]:
                array_data[j][0], array_data[j - 1][0] = array_data[j - 1][0], array_data[j][0]
                array_data[j][1], array_data[j - 1][1] = array_data[j - 1][1], array_data[j][1]
            else:
                pass
    for i in range(top_k, len(array_data)):
        if array_data[i][0] > array_data[top_k - 1][0]:
            array_data[top_k - 1][0] = array_data[i][0]
            array_data[top_k - 1][1] = array_data[i][1]
            for j in range(top_k - 1, 0, -1):
                if array_data[j][0] > array_data[j - 1][0]:
                    array_data[j][0], array_data[j - 1][0] = array_data[j - 1][0], array_data[j][0]
                    array_data[j][1], array_data[j - 1][1] = array_data[j - 1][1], array_data[j][1]
                else:
                    pass


def cpu_find_most_probability_top_n_from_block_result(results, top_numb):
    """
    这是个有bug的算法。如果是个升序排列的数组，那么结果只有最大的值。

    :param results:
    :param top_numb:
    :return:
    """
    top_n_results = np.zeros((top_numb, 2), np.float32)
    for i in range(len(results)):
        probability = results[i][0]
        frame_index = results[i][1] + 1
        if probability > top_n_results[0][0]:
            top_n_results[0][0] = probability
            top_n_results[0][1] = frame_index
            continue
        else:
            if probability < top_n_results[-1][0]:
                continue
            else:
                for k in range(1, top_numb + 1):
                    if probability > top_n_results[-k][0]:
                        continue
                    top_n_results[-k + 1][0] = probability
                    top_n_results[-k + 1][1] = frame_index
                    break
    # print(top_n_results)
    return top_n_results


def find_most_probability_top_n_from_result(results, start_index, end_index, test_len, top_numb):
    """
    从result中，找到top N。根据test_len长度的一半去重。

    :param results: source feature s和 test features 依次比较的全部结果。
    :param start_index: result开始计算的index。
    :param end_index: result停止计算的index。
    :param test_len: 待检测视频的长度（帧数）。
    :param top_numb: 取前 top N 的结果。
    :return:
    """
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

    return top_results


def take_first(elem):
    return elem[0]


def find_most_probability_top_n_from_full_result(results, test_len, top_numb):
    return find_most_probability_top_n_from_result(results, 0, len(results), test_len, top_numb)


def cpu_find_most_probability_with_interval(results, test_len, top_numb):
    top_results = [(0.0, 0)] * top_numb
    interval = int(test_len/2)
    for i in range(len(results)):
        a = results[i]
        if a < top_results[-1][0]:
            continue
        else:
            same_flag = 0
            for num in range(top_numb):
                if i + 1 - top_results[num][1] < interval:
                    same_flag = 1
                    if top_results[num][0] < a:
                        top_results[num] = (a, i + 1)
                        top_results.sort(key=take_first, reverse=True)
                        break
                    else:
                        break
            if same_flag == 0:
                top_results[-1] = (a, i + 1)
                top_results.sort(key=take_first, reverse=True)

    return top_results
