import datetime
from numpy import float32
import numpy as np



def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def search_file(test, source):
    list_source = text_read(source)
    list_test = text_read(test)

    n = len(list_source)
    m = len(list_test)
    print('n=%d  m=%d' % (n, m))
    a = np.array([0] * (n - m), dtype=float32)
    time_start_cal = datetime.datetime.now()

    result = float32(0)
    result_frame = 0
    for i in range(n-m):
        key = float32(0)
        for j in range(m):
            s1 = list_source[i+j].split()
            t1 = list_test[j].split()
            s = (s1[1],s1[2],s1[3],s1[4])
            t = (t1[1],t1[2],t1[3],t1[4])
            if s[0] == t[0]:
                key = key + 3 - np.abs(((float32(s[1]) - float32(t[1])))) * 9
            elif s[0] == t[2]:
                key = key + 1 - np.abs(((float32(s[1]) - float32(t[3]))))
            elif s[2] == t[0]:
                key = key + 1 - np.abs(((float32(s[3]) - float32(t[1]))))
            elif s[2] == t[2]:
                key = key + 1 - np.abs(((float32(s[3]) - float32(t[3]))))
            print('j=%d k=%f' % (j, key))
        a[i] = float32(key / m / 8)
        print('i=%d  key=%f  r=%f' % (i, key, a[i]))
        if a[i] > result:
            result_frame = i+1
            result = a[i]    

    time_end_cal = datetime.datetime.now()
    print('匹配度概率为：%.2f%%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分 %d 秒处'%(result*100,result_frame,int(result_frame/300),(result_frame%300)/5))
    print('*** 匹配用时：%s' % (time_end_cal - time_start_cal))

    return result, result_frame


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # test0008_AIresult_top5.txt  ==>  面具战士26_AIresult_top5.txt
    source_file = '/home/ai/source/U2RoseBowl体育演唱会_AIresult_top5.txt'
    test_file = '/home/ai/test/test0002_AIresult_top5.txt'
    search_file(test_file, source_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
