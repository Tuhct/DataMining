import sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA

iris = load_wine()
if __name__ == '__main__':
    iris = sklearn.datasets.load_wine()
    print(iris)

    # data对应了样本的4个特征，150行4列
    print('>> shape of data:')
    print(iris.data.shape)

    # 显示样本特征的前5行
    print('>> line top 5:')
    print(iris.data[:5])

    # target对应了样本的类别（目标属性），150行1列
    print('>> shape of target:')
    print(iris.target.shape)

    # 显示所有样本的目标属性
    print('>> show target of data:')
    print(iris.target)

    R=np.array(iris.data)

    # 求每一列的平均值
    B = np.mean(R, axis=0, keepdims=True)
    print('>>average matrix')
    print(B)

    # 得到均值矩阵
    R = np.matrix(R) - np.matrix(B)
    print('after-ave top 5 matrix')
    print(R[:5])

    # 计算协方差矩阵
    R_cov = np.cov(R, rowvar=False)
    iris_covmat = pd.DataFrame(data=R_cov, columns=iris.feature_names)
    print('>>covmat')
    print(iris_covmat)

    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(R_cov)
    # 选取前两个特征向量进行后续操作
    featureVector = eig_vectors[:, :2]
    print('>>eig_values:')
    print(eig_values)
    print('>>eig_vectors:')
    print(eig_vectors)

    # 二维简化：
    featureVector_t = np.transpose(featureVector)
    R_t = np.transpose(R)
    newDataset_t = np.matmul(featureVector_t, R_t)
    newDataset = np.transpose(newDataset_t)
    print('>>2D data:')
    print(newDataset[:5])

    # 用sklearn进行PCA
    # 降维维度为2
    pca = PCA(n_components=2)
    pca.fit(iris.data)
    print('>>proportion:')
    print(pca.explained_variance_ratio_)
    print('>>variance of feature dimension：')
    print(pca.explained_variance_)
    R2_new = pca.transform(iris.data)
    print('>>top 5 after_dealt:')
    print(R2_new[:5])

