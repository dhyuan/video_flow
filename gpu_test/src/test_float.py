import numpy as np
from numpy import float32

if __name__ == '__main__':
    s = ('701', '0.071388', '712', '0.059103')
    t = ('593', '0.244063', '712', '0.074797')
    key = 0
    print(s, t)
    if s[0] == t[0]:
        key = key + 3 - abs(float(s[1]) - float(t[1])) * 9
    if s[0] == t[2]:
        key = key + 1 - abs(float(s[1]) - float(t[3]))
    if s[2] == t[0]:
        key = key + 1 - abs(float(s[3]) - float(t[1]))
    if s[2] == t[2]:
        key = key + 1 - abs(float(s[3]) - float(t[3]))
        print('%f', (float(s[3]) - float(t[3])))
        print('A: %f' % key)
    print('(i=1 j=10)  k=%f' % key)

    key = 0
    if s[0] == t[0]:
        key = key + 3 - np.abs((float32(s[1]) - float32(t[1]))) * 9
        print('A: %f' % key)
    if s[0] == t[2]:
        key = key + 1 - np.abs((float32(s[1]) - float32(t[3])))
    if s[2] == t[0]:
        key = key + 1 - np.abs((float32(s[3]) - float32(t[1])))
    if s[2] == t[2]:
        key = key + 1 - np.abs(((float32(s[3]) - float32(t[3]))))
        print('%f', (float(s[3]) - float(t[3])))
        print('A: %f' % key)
    print('(i=1 j=10)  k=%f' % key)
