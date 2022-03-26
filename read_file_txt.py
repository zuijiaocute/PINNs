"""
流程：
处理数据集：
难点      是一个 26 * 46 * 6米的仓库，已经划分成80 * 141 * 19 = 214320 个点
需要处理成 1920个点左右                  理论：16 * 30  * 4  = 1920   个点
这些点的值应该要满足边界的对应关系
时间上，有两年17520个时间点，每24个小时选一个点，选择730个点，用前723个时间预测最后7个点


1、一个数据集，只有题t = 0时刻的点，可以单独做一个数据集，只要从一个body_001中选出这些点就可以了
2、body_001、025每隔24个文件，选择一个文件，然后同样的选择这些点，这些制作成第二个数据集

3、数据集可以随机打乱，每次选择batch个点，直接带入相应的网络和方程，这个在pinn_torch里似乎已经有了
4、直接用之前的网络，但是要对应一下xyz轴对应的参数，同时要对比看为什么求导出问题答案是0

"""
import os
import numpy as np
import pandas as pd


class dataPreprocess():
    def __init__(self) -> None:
        pass

    @staticmethod
    def trans_txt_to_numpy(save_data_type):
        assert save_data_type == "float16" or save_data_type == "float32"
        if(save_data_type == "float16"):
            np_type = np.float16
        else:
            np_type = np.float32

        for i,j in enumerate(range(1,17520,24),1):
            print("Working on ",i,"/ 730!" )
            txt_name = "body-" + str(j).rjust(4,"0") + ".txt"
            data_path = os.path.join('/Volumes/Extreme SSD/gain_yu',txt_name)
            df =  pd.read_csv(data_path, sep = ', ')
            df = df.iloc[:,1:]
            data = df.to_numpy()
            data = data.astype(np_type)

            output_name = str(i) + '.npy'
            output_path = os.path.join('/Users/xlj/Desktop/grain_dataset',output_name)

            np.save(output_path,data)

        print("Finish!")

dataPreprocess.trans_txt_to_numpy("float32")

#df =  pd.read_csv('/Volumes/Extreme SSD/gain_yu/body-0001.txt', sep = ', ')

#测试文件占用大小 
# df =  pd.read_csv('/Users/xlj/desktop/body-0001.txt', sep = ', ')
# df = df.iloc[:,1:]
# data = df.to_numpy()
# data = data.astype(np.float16)
# print(data)
# np.save("./text.npy",data)
#print(txt001_dir)

# data = []
# for i in range(1,241):
#     file_add = './reslut_grain_full/point' + str(i) + '.out'
#     with open(file_add,'r') as f:
#         data_point = []
#         cnt = 0
#         for line in f:
#             cnt += 1
#             if(cnt <= 3): continue
#             string = line.split(' ')[1].strip('\n')
#             temp = float(string) - 273
#             data_point.append(temp)
#         data.append(data_point)

# np_data = np.array(data).reshape(10,6,4,104).transpose(2,0,1,3)
# print(np_data.shape)
# np.save("./np_4_10_6_104.npy",np_data)