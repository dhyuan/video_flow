import threading

import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import time

from concurrent.futures import ThreadPoolExecutor
from queue import Queue, PriorityQueue

model_path = os.path.join("./model", 'classify_image_graph_def.pb')


def create_graphs():
    graphs = []
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        for i in range(5):
            graph = tf.Graph()
            graph.ParseFromString(f.read())
            tf.import_graph_def(graph, name='graph_%s' % i)
            graphs[i] = graph
    return graphs


def create_default_graph():
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        for i in range(5):
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


if __name__ == '__main__':

    print(tf.__version__)
    print(tf.__path__)

    create_default_graph()

    def process_multi_imgs():
        images_data = get_list_of_image_data('./images/10')
        print('========images_data LEN:  %d ' % len(images_data))

        beg_time = datetime.now()
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
            print(softmax_tensor)
            for i in range(len(images_data)):
                time1 = datetime.now()
                image_data = images_data[i]
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                time2 = datetime.now()
                print('softmax_tensor计算时间为：%s' % (time2 - time2))
                predictions = np.squeeze(predictions[0])  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
                top5 = predictions.argsort()[-5:][::-1]
                time3 = datetime.now()
                print('图片预测计算时间为：%s' % (time3 - time1))

        end_time = datetime.now()
        print('Total_Time for one session: %s ' % (end_time - beg_time))


    def process_one_img():
        image_data = tf.gfile.FastGFile("./images/111.jpg", 'rb').read()
        print(type(image_data))
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
            time1 = datetime.now()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 输入feed_dict进行运算
            predictions = np.squeeze(predictions)  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
            top5 = predictions.argsort()[-5:][::-1]
            time3 = datetime.now()
            print('图片预测计算时间为：%s' % (time3 - time1))


    # process_one_img()


    def get_list_of_image_data(path):
        files = os.listdir(path)
        images_data = []
        for f in files:
            file_name = '%s/%s' % (path, os.path.basename(f))
            print(file_name)
            if not os.path.isfile(file_name) or not f.endswith(".jpg"):
                continue

            img_data = tf.gfile.FastGFile(file_name, 'rb').read()
            images_data.append(img_data)
        return images_data


    def process_multi_imgs():
        images_data = get_list_of_image_data('./images/10')
        print('========images_data LEN:  %d ' % len(images_data))

        beg_time = datetime.now()
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
            print(softmax_tensor)
            for i in range(len(images_data)):
                time1 = datetime.now()
                image_data = images_data[i]
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                time2 = datetime.now()
                print('softmax_tensor计算时间为：%s' % (time2 - time2))
                predictions = np.squeeze(predictions[0])  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
                top5 = predictions.argsort()[-5:][::-1]
                time3 = datetime.now()
                print('图片预测计算时间为：%s' % (time3 - time1))

        end_time = datetime.now()
        print('Total_Time for one session: %s ' % (end_time - beg_time))

    process_multi_imgs()

    thread_pool = ThreadPoolExecutor(4, thread_name_prefix='ImageProcessThread')
    IMAGE_DATA_QUEUE = Queue()


    class ImageProcessThread(threading.Thread):
        def __init__(self, tf_session, work_queue, softmax_tensor):
            threading.Thread.__init__(self)
            self.tf_session = tf_session
            self.work_queue = work_queue
            self.softmax_tensor = softmax_tensor

        def process(self):
            image_data = self.work_queue.get(block=False)

            time1 = datetime.now()
            predictions = self.tf_session.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            time2 = datetime.now()
            print('softmax_tensor计算时间为：%s' % (time2 - time2))

            predictions = np.squeeze(predictions[0])  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
            top5 = predictions.argsort()[-5:][::-1]
            time3 = datetime.now()
            print('图片预测计算时间为：%s' % (time3 - time1))

        def run(self):
            while IMAGE_DATA_QUEUE.qsize() > 0:
                thread_pool.submit(self.process)


    def process_multi_imgs_thread_pool():
        images_data = get_list_of_image_data('./images/1001')
        print('========images_data LEN:  %d ' % len(images_data))

        beg_time = datetime.now()
        for img in images_data:
            IMAGE_DATA_QUEUE.put(img)
        print('Queue LEN:  %d ' % IMAGE_DATA_QUEUE.qsize())

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
            dispatcher = ImageProcessThread(sess, IMAGE_DATA_QUEUE, softmax_tensor)
            dispatcher.start()
            while IMAGE_DATA_QUEUE.qsize() > 0:
                time.sleep(20)

        print('Queue LEN:  %d ' % IMAGE_DATA_QUEUE.qsize())
        end_time = datetime.now()
        print('Total_Time for one session: %s ' % (end_time - beg_time))


    # process_multi_imgs_thread_pool()

