import sys
import datetime
import numpy as np


def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def search_file(test, source):
    print('\n%s --> %s' % (test, source))

    list_source = text_read(source)
    list_test = text_read(test)

    n = len(list_source)
    m = len(list_test)
    print('n=%d  m=%d' % (n, m))
    a = np.array([0] * (n - m), dtype=np.float32)
    time_start_cal = datetime.datetime.now()

    top = 10
    result = [0.0] * top
    result_frame = [0.0] * top
    for i in range(n-m):
        key = 0
        for j in range(m):
            s1 = list_source[i+j].split()
            t1 = list_test[j].split()
            s = (s1[1],s1[2],s1[3],s1[4])
            t = (t1[1],t1[2],t1[3],t1[4])
            if s[0] == t[0]:
                key = key + abs(3 - abs((float(s[1]) - float(t[1]))) * 5)
            if s[0] == t[2]:
                key = key + 1 - abs((float(s[1]) - float(t[3])))
            if s[2] == t[0]:
                key = key + 1 - abs((float(s[3]) - float(t[1])))
            if s[2] == t[2]:
                key = key + 1 - abs((float(s[3]) - float(t[3])))
        a = key / m / 4
        # print(a)
        if a > result[0]:
            result_frame[0] = i + 1
            result[0] = a
            continue
        else :
            if a < result[-1]:
                continue
            else:
                for k in range(1,top+1):
                    if a > result[-k]:
                        continue
                    result_frame[-k+1] = i + 1
                    result[-k+1] = a
                    break

    time_end_cal = datetime.datetime.now()
    print('*** 匹配用时：%s' % (time_end_cal - time_start_cal))
    print('源： %s 的匹配情况为：'%source)
    for i in range(top):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处'%(i+1,result[i]*100,result_frame[i],int(result_frame[i]/300),(result_frame[i]%300)/5))

    return result, result_frame


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        source_file = sys.argv[2]
        search_file(test_file, source_file)
    else:
        print("-")

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
