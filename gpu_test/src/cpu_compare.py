import sys
import datetime



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
    a = [0] * (n - m)
    time_start_cal = datetime.datetime.now()

    result = 0
    result_frame = 0
    for i in range(n-m):
        key = 0
        for j in range(m):
            s1 = list_source[i+j].split()
            t1 = list_test[j].split()
            s = (s1[1],s1[2],s1[3],s1[4])
            t = (t1[1],t1[2],t1[3],t1[4])
            if s[0] == t[0]:
                key = key + 3 - ((float(s[1]) - float(t[1]))) * 9
            if s[0] == t[2]:
                key = key + 1 - ((float(s[1]) - float(t[3])))
            if s[2] == t[0]:
                key = key + 1 - ((float(s[3]) - float(t[1])))
            if s[2] == t[2]:
                key = key + 1 - ((float(s[3]) - float(t[3])))
        a[i] = key / m / 8
        if a[i] > result:
            result_frame = i+1
            result = a[i]    

    time_end_cal = datetime.datetime.now()
    print('匹配度概率为：%.2f%%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分 %d 秒处'%(result*100,result_frame,int(result_frame/300),(result_frame%300)/5))
    print('*** 匹配用时：%s' % (time_end_cal - time_start_cal))

    return result, result_frame


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
        test_file = sys.argv[2]
        search_file(test_file, source_file)
    else:
        print("-")

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
