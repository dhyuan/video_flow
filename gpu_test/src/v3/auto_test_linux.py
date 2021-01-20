#!/usr/bin/env python
#coding:utf-8
import os, sys, getopt
import numpy as np
import tensorflow as tf
import time
import glob
import subprocess,sys
import socket
import ftp_client
from gettest import *
from getsource import *
from update_datebase_feature import *
import MyDatabase
from datetime import datetime
from numba import cuda, jit
import math


# 准备目录
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(dir_path, 'saved_model')
model_path = os.path.join(model_dir, 'classify_image_graph_def.pb')

def create_graph():
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# 执行预测
def predict_image(result_file):
    time1=time.time()
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()  # 读取图片
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 输入feed_dict进行运算
        predictions = np.squeeze(predictions)  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
        top5 = predictions.argsort()[-5:][::-1]
        time3=time.time()
        printlog('图片预测计算时间为：'+str(round(time3-time1,2))+'s')
        # printlog('预测结果是：')
        result = [0]*5
        result_score = [0]*5
        i = 0
        for node_id in top5:
            score = predictions[node_id]
            # print('node_id= %d,score= %f'%(node_id,score))
            if node_id == 1000:
                node_id = 0
            node_id = str(node_id)
            while(len(node_id)<3):
                node_id = '0'+node_id
            # print(str(node_id)+'\n')
            result[i]=node_id
            result_score[i]=score
            i+=1
        printlog('最终结果为：%s %f  %s %f  %s %f  %s %f  %s %f \n'%(result[0],result_score[0],result[1],result_score[1],result[2],result_score[2],result[3],result_score[3],result[4],result_score[4]))
        f1 = open(result_file, 'a+')
        f1.write('%s %s %f  %s %f  %s %f  %s %f  %s %f \n'%(image,result[0],result_score[0],result[1],result_score[1],result[2],result_score[2],result[3],result_score[3],result[4],result_score[4]))
        f1.close()

def printlog(str):
    print(str)
    f = open(log_file, 'a+')
    f.write(str+'\n')
    f.close()
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
        time_end_gpu_cal = datetime.now()
        # print('*** GPU计算用时：%s' % (time_end_gpu_cal - time_begin_gpu_cal))

        #log_data(file_name, file_feature_dict[file_name], h_results, source_len, source_features_len)

        ts_top10_result = find_most_probability(h_results, test_len,top_numb)
        print_top10(ts_top10_result, top_numb)

        for i in range(top_numb):
            one_result = (ts_top10_result[i][0], ts_top10_result[i][1], file_name)
            final_result.append(one_result)
        final_result.sort(key=take_first, reverse=True)
        final_result = final_result[0:top_numb:1]

    print_final_top10(final_result, test_file, top_numb)

    end_time = datetime.now()
    # print('\n在 %d 个文件中比对 %s 用时: %s' % (sources_numb, test_file, (end_time - begin_compare_time)))
    # print('加载原特征库用时: %s' % (end_load_source_time - start_time))
    # print("\nTotalTime: %s" % (end_time - start_time))
    return


def log_data(file_name, src_features, h_results, source_len, source_features_len):
    print('RL file: %s  source_len=%d' % (file_name, source_len))
    # for i in range(source_len):
    #     print('\n%d  %f, %f, %f, %f' % (i, src_features[i][0], src_features[i][1], src_features[i][2], src_features[i][3]))
    #     print('%f' % (h_results[i]))

def print_final_top10(top10_result, file_name, top_numb):
    print('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：' % file_name)
    (no_use, test_result_file_name) = os.path.split(file_name)
    (test_result_file_name_true, extension) = os.path.splitext(test_result_file_name)
    f = open(result_dir + '/' + test_result_file_name_true + '_GPU_search_result.txt', 'a+')
    # f.write('\n\n-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：\n' % file_name)
    for i in range(top_numb):
        print('top%d：  概率： %.2f%% ,  源： %s , 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' % (
            i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
            (top10_result[i][1] % 300) / 5))
        f.write('top%d：  概率： %.2f%% ,  源： %s , 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处\n' % (
            i + 1, top10_result[i][0] * 100, top10_result[i][2], top10_result[i][1], int(top10_result[i][1] / 300),
            (top10_result[i][1] % 300) / 5))
    f.close()


def take_first(elem):
    return elem[0]


def print_top10(ts_top10_result, top_numb):
    for i in range(top_numb):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处' %
              (i + 1, ts_top10_result[i][0] * 100, ts_top10_result[i][1], int(ts_top10_result[i][1] / 300), (ts_top10_result[i][1] % 300) / 5))


def find_most_probability(results, test_len, top_numb):
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
        if file.startswith('.') or os.path.getsize(file) == 0:
            print('忽略空文件: %s' % file)
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

def update_source_result():
    get_source = MySource()
    database_source_result = get_source.getAllOkIds(0,9999)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hu:", ["help", "file="])
    print(args[0])
    IS_WIN32 = 'win32' in str(sys.platform).lower()
    if IS_WIN32:
        source_path = 'D:\\tempor\\ftptest'
        image_root_path = 'D:\\tempor\\ftptest'
        resule_path = 'D:\\tempor\\ftptest'
        log_path = 'D:\\tempor\\ftptest\\log'       # 提取特征值log存放路径

        sources_dir = 'D:\\tempor\\ftptest\\source_result'  # 比对 所有源的特征值文本 存放路径
        result_dir = 'D:\\tempor\\ftptest\\result-log'                  # 比对log存放路径

    else:
        source_path = '/storage/auto_test/tmp/'
        image_root_path = '/storage/auto_test/tmp/'
        resule_path = '/storage/auto_test/tmp/'
        log_path = '/storage/auto_test/tmp/'

        sources_dir = '/storage/auto_test/source_result/'  # 比对 所有源的特征值文本 存放路径
        result_dir = '/storage/auto_test/result_log/'  # 比对log存放路径
    ftp_root_path = '/local-storage/test/'

    update_database_source_feature(sources_dir)
    d_sources_features_on_gpu, file_feature_dict = load_features_to_gpu(sources_dir)

    create_graph()
    get_test = MyTest(args[0])

    while True:
        source = get_test.getOne()
        if not source  or source['id'] == None:
            time.sleep(30)
            continue

        id = source['id']
        remote_filepath = source['filepath']
        remote_filepath = remote_filepath.replace('\\','/')
        # remotefile = ftp_root_path + remote_filepath
        remotefile = os.path.join(ftp_root_path,remote_filepath)
        print(remotefile)

        (remote_path, extension) = os.path.splitext(remotefile)
        print(remote_path,extension)

        if IS_WIN32:
            filename = remotefile.split("\\")[-1]
        else:
            filename = remotefile.split("/")[-1]
        (filepath1, filename) = os.path.split(remotefile)
        print('----------------------------------------------------'+filename)

        print(filename)
        (filename_no_extension, extension) = os.path.splitext(filename)
        print('------source_path 为： %s'% source_path )
        # localfile = os.path.join(source_path, filename_no_extension)
        localfile = source_path + '/' + filename
        log_file = os.path.join(log_path, filename_no_extension + '_analysis_log.txt')
        a = 1
        while os.path.exists(log_file):
            log_file = os.path.join(log_path, filename_no_extension + '_名称重复_' + str(a) + '_analysis_log.txt')
            a = a + 1

        print('----------------------localfile 为：------------------------'+localfile)

        downloaded = 0
        for i in range(10):
            try:
                ftp_client.download(remotefile,localfile)
                printlog('-----文件 %s 下载完成--------'%localfile)
                downloaded = 1
                break
            except:
                printlog('-----文件 %s 下载 失败--------'%localfile)
                time.sleep(60)

        if downloaded == 0:
            continue

        if IS_WIN32:
            image_file_path = image_root_path.rstrip('/') + os.sep +filename_no_extension
            i = 1
            while os.path.exists(image_file_path):
                image_file_path = image_root_path.rstrip('/')+ os.sep + filename_no_extension + '_名称重复_' + str(i)
                i = i + 1
            os.makedirs(image_file_path)
            printlog('图片存放位置为：%s' % image_file_path)
            cmd = 'ffmpeg -i %s -r 5 -qscale:v 2 %s' % (localfile, image_file_path) + '\\%5d.jpg'
            printlog(cmd)
            os.system(cmd)
            path_file_num = glob.glob(image_file_path + '\\*.jpg')
        else:
            image_file_path = image_root_path.rstrip('/') + os.sep+ remote_path
            i = 1
            while os.path.exists(image_file_path):
                image_file_path = image_root_path.rstrip('/') + os.sep+ remote_path + '_' + str(i)
                i = i + 1
            os.makedirs(image_file_path)                                   # 当文件名最后面有空格时，生成的文件夹也带着空格
            printlog('图片存放位置为：%s' % image_file_path)
            cmd = '/root/ffmpeg -i \"%s\" -r 5 -qscale:v 2 \"%s\"' % (localfile, image_file_path) + '/%5d.jpg'        # 输入输出都必须带着引号，防止文件名里有空格
            printlog(cmd)
            os.system(cmd)
            path_file_num = glob.glob(image_file_path + '/*.jpg')

        printlog('文件 %s 抽帧结束，开始AI特征值分析......' % filename)
        num = len(path_file_num)
        printlog('图片总数为：' + str(num) + '张')

        result_file = os.path.join(resule_path, filename_no_extension + '_AIresult_top5.txt')
        a = 1
        while os.path.exists(result_file):
            result_file = os.path.join(resule_path, filename_no_extension + '_名称重复_' + str(a) + '_AIresult_top5.txt')
            a = a + 1

        for i in range(num):
            i = str(i + 1)
            while (len(i) < 5):
                i = '0' + i
            image = str(i) + '.jpg'
            image_path = os.path.join(image_file_path,image)
            printlog(image_path)

            a = predict_image(result_file)

        test_file = result_file

        # test_dir = 'D:\\AI\\result-test'
        # test_files = get_files_from_dir(test_dir)
        compare(test_file, d_sources_features_on_gpu, file_feature_dict, 10)
        # f = open(result_dir + '\\' + test_file + '_GPU_search_result.txt', 'a+')
        (no_use, test_result_file_name) = os.path.split(test_file)
        (test_result_file_name_true, extension) = os.path.splitext(test_result_file_name)
        test_file_compare_result = result_dir + '/' + test_result_file_name_true + '_GPU_search_result.txt'

        os.remove(localfile)
        get_test.finishOneFile(id,resultfile=test_file_compare_result)
