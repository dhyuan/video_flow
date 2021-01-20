import datetime


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
    a = [0] * (n - m)
    result = 0
    result_frame = 0
    time_start_cal = datetime.datetime.now()
    for i in range(n - m):
        key = 0
        for j in range(m):
            s = list_source[i + j].split()
            t1 = list_test[j].split()
            t = (t1[1], t1[2], t1[3], t1[4])
            if s[0] == t[0]:
                key = key + 6 - abs((float(s[1]) - float(t[1]))) * 18
            if s[0] == t[2]:
                key = key + 2 - abs((float(s[1]) - float(t[3]))) * 2
            if s[2] == t[0]:
                key = key + 2 - abs((float(s[3]) - float(t[1]))) * 2
            if s[2] == t[2]:
                key = key + 2 - abs((float(s[3]) - float(t[3]))) * 2

        a[i] = key / m / 8
        if a[i] > result:
            result_frame = i + 1
            result = a[i]
    time_end_cal = datetime.datetime.now()
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))
    print('*** 匹配用时：%s' % (time_end_cal - time_start_cal))

    return result, result_frame


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    source_files = ['./data/冰上恋人_01.txt']
    for src_file in source_files:
        search_file('./data/test008_AIresult_top5.txt', src_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))
