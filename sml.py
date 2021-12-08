# -*- coding: utf-8 -*-
'''
# @ File    :   sml.py
# @ Time    :   2021/12/03
# @ Author  :   k-starmon
# @ Version :   1.0
# @ Contact :   1993371310@qq.com
'''

import time
import pickle
import cv2 as cv
import numpy as np
from sklearn.mixture import GaussianMixture

''' define  : zigzag 扫描
    input   : 二维矩阵, shape: (row, col)
    output  : 列表, shape: (row*col,)
    variable: k 列表序号, i 行序号, j 列序号, row 行数, col 列数
    method  : 假设 (0, 0) 在左上角, (row-1, col-1) 在右下角的情况. 考虑非边界的情况, 只有右上/左下两个方向.
              以从 (0, 0) 先向右(下)为例, 则会有 i+j 为偶数时右上(左下)前进, 为奇数时左下(右上)的情况前进.
              如果遇到边界, 某个方向收到限制, 移动允许的直线方向'''
def zigzag(data):
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    list = np.zeros(num,)
    k = 0
    i = 0
    j = 0

    while i < row and j < col and k < num:
        list[k] = data.item(i, j)
        k = k + 1
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            if (i-1) in range(row) and (j+1) not in range(col):
                i = i + 1            # 右边界超出, 则向下
            elif (i-1) not in range(row) and (j+1) in range(col):
                j = j + 1            # 上边界超出, 则向右
            elif (i-1) not in range(row) and (j+1) not in range(col):
                i = i + 1            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            if (i+1) in range(row) and (j-1) not in range(col):
                i = i + 1            # 左边界超出, 则向下
            elif (i+1) not in range(row) and (j-1) in range(col):
                j = j + 1            # 下边界超出, 则向右
            elif (i+1) not in range(row) and (j-1) not in range(col):
                j = j + 1            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            else:
                i = i + 1
                j = j - 1
    
    return list


''' define  : 加载一个语义类的所有图片并转为YCrCb格式
    input   : path-图片所在文件夹的路径, image_num-图片数量, , row_pixel-行像素点, col_pixel-列像素点
    output  : YBR 图片, [image_num, row_pixel, col_pixel, 3], 3-通道数'''
def load_picture(path, image_num, row_pixel, col_pixel):
    # 初始化
    img = np.zeros((image_num, row_pixel, col_pixel, 3))  
    # 遍历文件夹里所有图片
    for i in range(image_num):
        # opencv 读取数据建议路径不要出现中文
        temporary_img = cv.imread(path + str(i + 1) + '.jpg')
        # 转　BGR　为　YCrCb
        img[i, :, :, :] = cv.cvtColor(temporary_img, cv.COLOR_BGR2YCrCb)
    return img


''' define  : 分割图片并转换为dct, 将每一个小块的数据展为一维
    input   : [10, 300, 450, 3], 依次为图片序号、纵横像素点、通道
    output  : [10, split-num, 8*8*3] 格式的图片, 依次为图片序号、块序号、块列表'''
def split_and_2dct(img):
    # 计算该分成多少块，8*8大小，重叠2
    row_num = int((img.shape[1] - 2) / 6)
    col_num = int((img.shape[2] - 2) / 6)
    num = col_num * row_num

    split_dct_img = np.zeros((img.shape[0], num, 8*8*3))
    # 第 j 张图片
    for j in range(img.shape[0]):
        # 切割第 i 个小块
        for i in range(num):
            row = (int(i / row_num))*6      # 计算行的像素起始点
            col = (i % row_num)*6           # 计算列的像素起始点
            temporary_img = img[j, row:row + 8, col:col + 8, :]
            temporary_img_list = np.zeros((8*8, 3))
            # 对不满 8*8 的图片块进行补0
            if temporary_img.shape[0] != 8:
                add_sum = 8 - temporary_img.shape[0]
                temporary_img = np.pad(temporary_img, ((0, add_sum), (0, 0), (0, 0)), 'constant')
            if temporary_img.shape[1] != 8:
                add_sum = 8 - temporary_img.shape[1]
                temporary_img = np.pad(temporary_img, ((0, 0), (0, add_sum), (0, 0)), 'constant')
            # 转换数据类型, 之后就可以用 opencv 进行 dct 变换
            temporary_img = temporary_img.astype(np.float32)
            # 对每个通道进行 dct 变换, 变化区域的大小为 8*8, 然后用 zigzag 转为一维
            for k in range(3):
                temporary_img[:, :, k] = cv.dct(temporary_img[:, :, k])
                temporary_img_list[:, k] = zigzag(temporary_img[:, :, k])
            # 将 (64, 3) 的数据降维, YBR 交替出现
            split_dct_img[j, i, :] = temporary_img_list.T.reshape(1, 8*8*3)
    return split_dct_img


''' define  : 进行图像特征提取, 基于 6 组高斯分量
    input   : [image_num, split-num, 8*8*3] 格式的图片, 依次为图片序号、块序号、块列表, 8*8*3 = 192
    output  : [image_num*feature_num, 1+192+192] 一个语义类的 Gaussian 参数样本, 1+192+192 = 385
    explain : 对于每幅图片的 split-num 个分块, 每个分块作为一个样本, 进行 6 个分量的聚类,
              将如此聚类得到的参数进行整合, 作为一组数据, 对于一个语义类, 可得到 10*6 个样本数据'''
def feature_GMM(img, feature_num):
    # 初始化样本数据, 10 张图片各 6 组高斯参数
    image_feature = np.zeros((img.shape[0]*feature_num, 1+192+192))
    for k in range(img.shape[0]):
        # 初始化一张图片的 6 组高斯参数
        # 混合高斯模型个数 6 , 模型初始化参数的方式 k-means, 协方差类型 每个分量有各自不同对角协方差矩阵
        gmm = GaussianMixture(n_components = feature_num, init_params = 'kmeans', covariance_type = 'diag').fit(img[k, :, :])
        weight = gmm.weights_               # 每个混合模型的权重, 即pi, 维度 (6,)
        mean = gmm.means_                   # 每个混合模型的均值, 即mu, 维度 (6,192)
        covariances = gmm.covariances_      # 每个混合模型的协方差, 即sigma, 维度 (6,192)
        # 将 pi, mu, sigma 按列拼接
        image_feature[feature_num*k:feature_num*k+feature_num] = np.c_[weight, mean, covariances]
    return image_feature


''' define  : 主函数, 预测'''
def sml():

    # paras
    data_path = './SML/dataset/'        # 数据集路径
    model_path = './SML/model/'         # 训练集路径
    row_pixel = 300                     # 图片像素-行
    col_pixel = 450                     # 图片像素-列
    my_class = 4                        # 语义类个数
    image_num = 10                      # 每个语义类的图片个数
    test_num = 5                        # 测试集里的图片个数
    feature_num = 6                     # 图片特征的高斯分量个数
    class_num = 10                      # 类模型的高斯分量个数
    label = {   "1":"建筑", "2":"森林", 
                "3":"天空", "4":"道路"}  # 标签

    # 加载图片
    print('Load picture ...')
    start_time = time.time()
    class_1_img = load_picture(data_path + 'w1/', image_num, row_pixel, col_pixel)
    class_2_img = load_picture(data_path + 'w2/', image_num, row_pixel, col_pixel)
    class_3_img = load_picture(data_path + 'w3/', image_num, row_pixel, col_pixel)
    class_4_img = load_picture(data_path + 'w4/', image_num, row_pixel, col_pixel)
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - start_time))

    # 图片分块和dct
    print('Split picture and do the DCT transform ...')
    start_time = time.time()
    split_dct_img_1 = split_and_2dct(class_1_img)
    split_dct_img_2 = split_and_2dct(class_2_img)
    split_dct_img_3 = split_and_2dct(class_3_img)
    split_dct_img_4 = split_and_2dct(class_4_img)
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - start_time))

    # 特征抽取
    print('Image feature extraction using GMM method ...')
    start_time = time.time()
    image_feature_1 = feature_GMM(split_dct_img_1, feature_num)
    image_feature_2 = feature_GMM(split_dct_img_2, feature_num)
    image_feature_3 = feature_GMM(split_dct_img_3, feature_num)
    image_feature_4 = feature_GMM(split_dct_img_4, feature_num)
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - start_time))

    # 类模型
    print('Class model processing using GMM method ...')
    start_time = time.time()
    class_params_1 = GaussianMixture(n_components = class_num, init_params = 'kmeans', covariance_type = 'diag').fit(image_feature_1)
    class_params_2 = GaussianMixture(n_components = class_num, init_params = 'kmeans', covariance_type = 'diag').fit(image_feature_2)
    class_params_3 = GaussianMixture(n_components = class_num, init_params = 'kmeans', covariance_type = 'diag').fit(image_feature_3)
    class_params_4 = GaussianMixture(n_components = class_num, init_params = 'kmeans', covariance_type = 'diag').fit(image_feature_4)
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - start_time))
    print('You have got the class model!')

    # 测试或者标注
    print('Test model ...')
    start_time = time.time()
    test_img = load_picture(data_path + 'test/', test_num, row_pixel, col_pixel)
    split_dct_test_img = split_and_2dct(test_img)
    test_image_feature = feature_GMM(split_dct_test_img, feature_num)
    print('标注方式：6 组高斯分量表示的特征, 在各个模型下对数似然求和')
    result = np.zeros(my_class,)
    for i in range(test_num):
        # 对于一个测试图片, 在每个语义类模型中进行评价, 并挑选对数似然最大的那个作为标注
        result1 = class_params_1.score_samples(test_image_feature[i*feature_num: i*feature_num+feature_num]).sum()
        result2 = class_params_2.score_samples(test_image_feature[i*feature_num: i*feature_num+feature_num]).sum()
        result3 = class_params_3.score_samples(test_image_feature[i*feature_num: i*feature_num+feature_num]).sum()
        result4 = class_params_4.score_samples(test_image_feature[i*feature_num: i*feature_num+feature_num]).sum()
        result = np.c_[result1, result2, result3, result4]
        # 获取最大值的索引
        maxindex  = np.argmax(result)
        print('第' + str(i+1) + '张图片的标注：', label[str(maxindex+1)])
        print('模型一', result1)
        print('模型二', result2)
        print('模型三', result3)
        print('模型四', result4)
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - start_time))

    # # 保存模型, 已经保存, 不用再运行
    # print('Save model ...')
    # start_time = time.time()
    # with open (model_path + '1.pickle', 'wb') as f:
    #     pickle.dump(class_params_1, f)
    # with open (model_path + '2.pickle', 'wb') as f:
    #     pickle.dump(class_params_2, f)
    # with open (model_path + '3.pickle', 'wb') as f:
    #     pickle.dump(class_params_3, f)
    # with open (model_path + '4.pickle', 'wb') as f:
    #     pickle.dump(class_params_4, f)
    # end_time = time.time()
    # print('Running time: %s Seconds' % (end_time - start_time))

if __name__ == "__main__":
    sml()