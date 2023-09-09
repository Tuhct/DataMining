import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy as np

path = "D:\Python\数据挖掘\winequality-red.csv"
wine = pd.read_csv(path, sep=';', header=0, encoding='utf-8')
# print(wine.shape)
wine = wine.drop_duplicates()
# print(wine.info())
# print(wine.describe())
# print(wine['quality'].value_counts())
'''
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
CORR=wine.corr(method='pearson')
print(CORR)'''
'''
plt.scatter(x=wine.alcohol,y=wine.quality)
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.savefig('./img/1.png')
plt.show()'''

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 10)
bin = (2, 4, 6, 8)
lablename = ['low', 'medium', 'high']
wine['level'] = pd.cut(x=wine.quality, bins=bin, labels=lablename)
# print(wine.level.value_counts())

a = LabelEncoder()
wine['level_num'] = a.fit_transform(wine.level)
# print(wine)

'''
copy = wine.copy()
copy.drop(['quality', 'level','level_num'], axis=1, inplace=True)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print(copy)
copy=pca.fit_transform(copy)
print(copy)'''

'''
print('>>proportion:')
print(pca.explained_variance_ratio_)
print('>>variance of feature dimension：')
print(pca.explained_variance_)
R2_new = pca.transform(copy)
print('>>top 5 after_dealt:')
print(R2_new[:5])'''

copy2 = wine.copy()
a, b = np.split(copy2, indices_or_sections=(3,), axis=1)
# print(a)
x = a.iloc[:, :-1]
y = wine.level_num
#print(x)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.75, test_size=0.25)
# print(y_test.shape)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
classifier = svm.SVC(C=5, kernel='rbf', gamma=20, decision_function_shape='ovr')
classifier.fit(x_train, y_train.ravel())
print("Training_set_score：", format(classifier.score(x_train, y_train), '.3f'))
print("Testing_set_score：", format(classifier.score(x_test, y_test), '.3f'))

i=x['fixed acidity']
#print(i)
j=x['volatile acidity']
#print(j)
# 第1维特征
x1_min = i.min()
x1_max = i.max()
print(x1_max)
print(x1_min)
# 第2维特征
x2_min = j.min()
x2_max = j.max()
print(x2_max)
print(x2_min)
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['pink', 'skyblue', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
plt.pcolormesh(x1, x2, grid_hat,shading='auto', cmap=cm_light)     # 预测值的显示
plt.scatter(i, j, c=y, s=30,cmap=cm_dark)  # 样本
plt.scatter(x_test['fixed acidity'],x_test['volatile acidity'], c=y_test,s=30,edgecolors='k', zorder=2,cmap=cm_dark) #圈中测试集样本点
plt.xlabel('fixed acidity', fontsize=13)
plt.ylabel('volatile acidity', fontsize=13)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title('wine 数据集三维图像')
plt.show()
