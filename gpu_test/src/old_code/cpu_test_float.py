import datetime



def text_read(f):
    file = open(f, 'r')
    lines = file.readlines()
    return lines


def to_tuple(data):
    ds = data.split()
    return int(ds[0]), float(ds[1]), int(ds[2]), float(ds[3])


def search_file(test, source):
    list_source = text_read(source)
    list_test = text_read(test)

    src_data = list(map(lambda d: to_tuple(d), list_source))
    test_data = list(map(lambda d: to_tuple(d), list_test))

    n = len(src_data)
    m = len(test_data)
    print('n=%d  m=%d' % (n, m))
    a = [0] * (n - m)
    result = 0
    result_frame = 0

    time_start_cal = datetime.datetime.now()
    for i in range(n - m):
        for j in range(m):
            s = src_data[i + j]
            t = test_data[j]
            if s[0] == t[0]:
                a[i] = a[i] + (s[1] + t[1])
            elif s[0] == t[2]:
                a[i] = a[i] + (s[1] + t[3])
            elif s[2] == t[0]:
                a[i] = a[i] + (s[3] + t[1])
            elif s[2] == t[2]:
                a[i] = a[i] + (s[3] + t[3])
        if a[i] > result:
            result_frame = i + 1
            result = a[i]
    time_end_cal = datetime.datetime.now()
    print('匹配度概率为: %.2f %%, 匹配位置在抽帧样本的第 %d 帧,即原视频 %d 分处' % (result * 100, result_frame, (result_frame % 300)))
    print('*** 匹配用时：%s' % (time_end_cal - time_start_cal))
    # assert(result == 22.360765)
    # assert(result_frame == 4029)

    return result, result_frame


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    source_files = ['./data/冰上恋人_01.txt']
    # source_files = ['./data/一公升的眼泪_log.txt']
    for src_file in source_files:
        search_file('./data/test008_AIresult_top5.txt', src_file)
        # search_file('./data/test1_log.txt', src_file)

    end_time = datetime.datetime.now()
    print("\n\nTotalTime: %s" % (end_time - start_time))

