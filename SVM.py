'''
iris = load_iris()
if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()

    print(iris[0:2,:])
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
'''
from sklearn.datasets import load_wine
from sklearn import svm
import matplotlib
import numpy as np
#绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
wine = load_wine() #得到数据

data=np.zeros((178, 2), dtype=float)

for i in range(0,len(wine.data)):
    for j in range(0,2):
        data[i][j]= wine.data[i][j]
        #print(wine.data[i][j])
#print(data)
wine.data=data
print(wine)
x=wine.data
y=wine.target
train_data,test_data,train_label,test_label=train_test_split(x,y,random_state=1,train_size=0.75,test_size=0.25)
'''
wine_target = wine.target #得到数据对应的标签，其中0，1，2分别代表'setosa', 'versicolor', 'virginica'三种不同花的类别。
wine_features = pd.DataFrame(data=data.data, columns=data.feature_names)
#data.data得到鸢尾花的数据（也就是花萼和花瓣各自的长宽）
#data.features_names得到data.data中各个数据的名称
#利用Pandas转化为DataFrame格式

# 利用info()查看数据的整体信息
wine_features.info()
print(wine_target)
'''
# 定义SVM分类器，希望大家还记得之前我们讲过的rbf核是什么

# C越大分类效果越好，但有可能会过拟合，gamma是高斯核参数，而后面的dfs制定了类别划分方式，ovr是一对多方式。
classifier = svm.SVC(C=5, kernel='rbf', gamma=20, decision_function_shape='ovr')
# 这里分类器的参数关系我会在实验报告中给出，比较复杂代码注释中不做详述
classifier.fit(train_data, train_label.ravel())  # 用训练集数据来训练模型。（ravel函数在降维时默认是行序优先）

# 计算svc分类器的准确率

print("Training_set_score：", format(classifier.score(train_data, train_label), '.3f'))
print("Testing_set_score：", format(classifier.score(test_data, test_label), '.3f'))
'''
# 绘制图形将实验结果可视化(注意现在我们就挑了前二维特征来画图，好画，事实上该数据集有四维特征呢，不好画)

# 首先确定坐标轴范围，通过二维坐标最大最小值来确定范围
# 第1维特征的范围（花萼长度）
x1_min = x[:, 0].min()
x1_max = x[:, 0].max()
# 第2维特征的范围（花萼宽度）
x2_min = x[:, 1].min()
x2_max = x[:, 1].max()
# mgrid方法用来生成网格矩阵形式的图框架
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点（其实是颜色区域），先沿着x1向右扩展，再沿着x2向下扩展
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 再通过stack()函数，axis=1，生成测试点，其实就是合并横与纵等于计算x1+x2
grid_value = classifier.predict(grid_test)  # 用训练好的分类器去预测这一片面积内的所有点，为了画出不同类别区域
grid_value = grid_value.reshape(x1.shape)  # （大坑）使刚刚构建的区域与输入的形状相同（裁减掉过多的冗余点，必须写不然会导致越界读取报错，这个点的bug非常难debug）
# 设置两组颜色（高亮色为预测区域，样本点为深色）
light_camp = matplotlib.colors.ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
dark_camp = matplotlib.colors.ListedColormap(['r', 'g', 'b'])
fig = plt.figure(figsize=(10, 5))  # 设置窗体大小
fig.canvas.set_window_title('SVM -2 feature classification of wine')  # 设置窗体title
# 使用pcolormesh()将预测值（区域）显示出来
plt.pcolormesh(x1, x2, grid_value, cmap=light_camp)
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=dark_camp)  # 加入所有样本点，以深色显示
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='white', zorder=2, cmap=dark_camp)
# 单独再把测试集样本点加一个圈,更加直观的查看命中效果
# 设置图表的标题以及x1,x2坐标轴含义
plt.title('SVM -2 feature classification of wine')
plt.xlabel('length of calyx')
plt.ylabel('width of calyx')
# 设置坐标轴的边界
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()'''