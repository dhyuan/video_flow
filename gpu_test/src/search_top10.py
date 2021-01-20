import os
import time
import glob

def text_read(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    return content

def searchfile(test, source,top):
    list_source = text_read(source)
    list_test = text_read(test)
    n = len(list_source)
    m = len(list_test)
    # a = [0]*(n-m)
    result = [0.0]*top
    result_frame = [0.0]*top
    # result = 0
    # result_frame = 0
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
        else:
            if a < result[-1]:
                continue
            else:
                for k in range(1,top+1):
                    if a > result[-k]:
                        continue
                    result_frame[-k+1] = i + 1
                    result[-k+1] = a
                    break

    f1 = open(log_path + '\\search_log.txt', 'a+')
    print('源： %s 的匹配情况为：'%source)
    f1.write('源： %s 的匹配情况为：\n'%source)
    for i in range(top):
        print('top%d：概率：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处'%(i+1,result[i]*100,result_frame[i],int(result_frame[i]/300),(result_frame[i]%300)/5))
        f1.write('top%d：概率为：%.8f%%, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处\n'%(i+1,result[i]*100,result_frame[i],int(result_frame[i]/300),(result_frame[i]%300)/5))
    f1.close()
    return (result,result_frame)

def takefirst(elem):
    return elem[0]


if __name__ == '__main__':
    log_path = "D:\\AI\\result-log"
    source_path = "D:\\AI\\result-source"
    source_list_all = os.listdir(source_path)
    source = []
    top = 10
    for file in source_list_all:
        (filename, extension) = os.path.splitext(file)
        if extension=='.txt':
            source.append(filename)
    print(source)

    test_path = "D:\\AI\\result-test"
    test_list_all = os.listdir(test_path)
    test = []
    for testfile in test_list_all:
        (filename, extension) = os.path.splitext(testfile)
        if extension == '.txt':
            test.append(filename)
    print(test)

    time1=time.time()
    for testfile in test:
        test = test_path + '\\' + testfile + '.txt'
        final_result = []
        tmp_result = []
        # final_place = ['']*top
        # final_source = ['']*top
        for w in range(len(source)):
            # lenTxt = len(source[w])
            # lenTxt_utf8 = len(source[w].encode('utf-8'))
            # size = int((lenTxt_utf8 - lenTxt) / 2 + lenTxt)
            # source_name = source[w] + " " * (40 - size)
            print('测试素材：%s 的匹配结果为：' %testfile,end="  ")
            f1 = open(log_path + '\\search_log.txt', 'a+')
            f1.write('测试素材：%s 的匹配结果为：' %testfile)
            f1.close()
            tmp_result_1 = searchfile(test,source_path+'\\'+source[w]+'.txt',top)
            # if tmp_result[0] > final_result:
            #     final_result = tmp_result[0]
            #     final_place = tmp_result[1]
            #     final_source = source[i]
            for i in range(top):
                a = (tmp_result_1[0][i],tmp_result_1[1][i],source[w])
                final_result.append(a)
            final_result.sort(key=takefirst,reverse = True)
            final_result = final_result[0:top:1]
            # print(final_result)


        print('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：'%testfile)
        f1 = open(log_path + '\\search_log.txt', 'a+')
        f1.write('-------------------------------------------------------------------------测试视频 %s 的最终匹配结果为：\n'%testfile)
        for i in range(top):
            print('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处'%(i+1,final_result[i][0]*100,final_result[i][2],final_result[i][1],int(final_result[i][1]/300),(final_result[i][1]%300)/5))
            f1.write('top%d：概率：%.2f%%, 源：%s, 抽帧样本 %d 帧,即原视频 %d 分 %d 秒处\n' % (i+1, final_result[i][0]*100,final_result[i][2],final_result[i][1],int(final_result[i][1]/300),(final_result[i][1]%300)/5))
        f1.write('\n\n')
        print('\n\n')
        f1.close()

        f = open(log_path + '\\search_result.txt', 'a+')
        f.write('%s %.2f %% %d 分 %d 秒\n'%(final_result[0][2],final_result[0][0]*100,int(final_result[0][1]/300),(final_result[0][1]%300)/5))
        f.close()


    time2=time.time()
    f = open(log_path + '\\search_result.txt', 'a+')
    f.write('共耗时：%.2f\n'%(time2-time1))
    f.close()
    print('共耗时：%.2f'%(time2-time1))


