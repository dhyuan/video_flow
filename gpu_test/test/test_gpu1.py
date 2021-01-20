from old_code.gpu_top5A_test import text_read
from old_code.gpu_top5A_test import trans_source_data_to_tuple
from old_code.gpu_top5A_test import trans_test_data_to_tuple
from old_code.gpu_top5A_test import compare_two_frame_feature


def test_float():
    s = ('701', '0.071388', '712', '0.059103')
    t = ('593', '0.244063', '712', '0.074797')
    key = 0
    print(s, t)
    if s[0] == t[0]:
        key = key + 3 - (float(s[1]) - float(t[1])) * 9
    if s[0] == t[2]:
        key = key + 1 - (float(s[1]) - float(t[3]))
    if s[2] == t[0]:
        key = key + 1 - (float(s[3]) - float(t[1]))
    if s[2] == t[2]:
        key = key + 1 - (float(s[3]) - float(t[3]))
    print('(i=1 j=10)  k=%f' % key)


def test_compare_two_frame_feature():
    source_feature = [752.0, 0.082397, 749.0, 0.058257]
    test_feature = [752.0, 0.082397, 749.0, 0.058257]
    acc = compare_two_frame_feature(source_feature, test_feature, 0)
    assert acc == 6

    acc = compare_two_frame_feature(source_feature, test_feature, 1.5)
    assert acc == 6 + 1.5


def test_trans_source_data_to_tuple():
    text = '752 0.082397  749 0.058257  548 0.057836  930 0.034443  917 0.029522\n'
    tuple_data = trans_source_data_to_tuple(text)
    assert tuple_data[0] == 752
    assert tuple_data[1] == 0.082397
    assert tuple_data[2] == 749
    assert tuple_data[3] == 0.058257


def test_trans_test_data_to_tuple():
    text = '00001.jpg 752 0.082397  749 0.058257  548 0.057836  930 0.034443  917 0.029522 \n'
    tuple_data = trans_test_data_to_tuple(text)
    assert tuple_data[0] == 752.0
    assert tuple_data[1] == 0.082397
    assert tuple_data[2] == 749
    assert tuple_data[3] == 0.058257


def test_read_source_data_from_file():
    text_lines = text_read('data/10_features_of_source.txt')

    assert len(text_lines) == 10
    assert text_lines[0].strip() == '749 0.158980  548 0.064519  752 0.037691  284 0.034389  842 0.027214'
    assert text_lines[1].strip() == '749 0.200537  752 0.055738  669 0.041539  548 0.038108  284 0.028101'
    assert text_lines[9].strip() == '930 0.142432  548 0.074825  749 0.058889  710 0.026414  961 0.022594'


def test_read_test_data_from_file():
    text_lines = text_read('data/3_features_of_test_at5.txt')

    assert len(text_lines) == 3
    assert text_lines[0].strip() == '00001.jpg 752 0.082397  749 0.058257  548 0.057836  930 0.034443  917 0.029522'
    assert text_lines[1].strip() == '00002.jpg 749 0.121634  930 0.077926  752 0.069406  548 0.043451  669 0.035839'
    assert text_lines[2].strip() == '00003.jpg 749 0.151907  930 0.084702  752 0.077291  548 0.076200  762 0.023804'
