import os
import numpy as np
dir = os.listdir('./reslut_grain_full/')


data = []
for i in range(1,241):
    file_add = './reslut_grain_full/point' + str(i) + '.out'
    with open(file_add,'r') as f:
        data_point = []
        cnt = 0
        for line in f:
            cnt += 1
            if(cnt <= 3): continue
            string = line.split(' ')[1].strip('\n')
            temp = float(string) - 273
            data_point.append(temp)
        data.append(data_point)

np_data = np.array(data).reshape(10,6,4,104).transpose(2,0,1,3)
print(np_data.shape)
np.save("./np_4_10_6_104.npy",np_data)