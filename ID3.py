from math import log


def creatDataSet():
    # 数据集
    labels = ["天气", "温度", "湿度", "风速", "活动"]
    dataSet = [["晴", "炎热", "高", "弱", "取消"]
        , ["晴", "炎热", "高", "强", "取消"]
        , ["阴", "炎热", "高", "弱", "进行"]
        , ["雨", "适中", "高", "弱", "进行"]
        , ["雨", "寒冷", "正常", "弱", "进行"]
        , ["雨", "寒冷", "正常", "强", "取消"]
        , ["阴", "寒冷", "正常", "强", "进行"]
        , ["晴", "适中", "高", "弱", "取消"]
        , ["晴", "寒冷", "正常", "弱", "进行"]
        , ["雨", "适中", "正常", "弱", "进行"]
        , ["晴", "适中", "正常", "强", "进行"]
        , ["阴", "适中", "高", "强", "进行"]
        , ["阴", "炎热", "正常", "弱", "进行"]
        , ["雨", "适中", "高", "强", "取消"]]
    return dataSet, labels

"""
计算熵
Ent=-∑pk*logpk 
"""
# 计算标准熵
def entropy(dataSet):
    # 返回数据集行数
    num_data = len(dataSet)
    ent = 0.0  # 经验熵
    # print(num_data)
    label_count = {}  # 保存每个标签（label）出现次数的字典
    # 对每组特征向量进行统计
    for i in dataSet:
        # 当使用负数索引时，python将从右开始往左数，因此 -1 是最后一个元素的位置
        label = i[-1]   # 提取标签信息
        if label not in label_count.keys():   # 如果标签没有放入统计次数的字典，添加进去
            label_count[label] = 0
        label_count[label]+=1  # label计数
        # print(label)
    # print(label_count)  # 输出每个标签（label）的出现次数
    # 计算经验熵
    for key in label_count.keys():
        Prob = float(label_count[key])/num_data  # 选择该标签的概率
        ent-=Prob*log(Prob,2)  #利用公式计算
    return ent  #返回经验熵

def splitDataSet(dataSet, axis, value):#划分子集，来求条件熵
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉axis特征
            reducedFeatVec = featVec[:axis]
            # print("reducedFeatVec",reducedFeatVec)
            # 将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis + 1:])
            # print("reducedFeatVec=",reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            # print("retDataSet=",retDataSet)
    # 返回划分后的数据集
    return retDataSet


"""
condition_Ent=∑(|D^v|/|D|)*Ent(D^v)
信息增益 Gain(D,a)=Ent(D)-∑(|D^v|/|D|)*Ent(D^v)
"""
def split_(dataSet, labels):
    num_data = float(len(dataSet))  # 数据集行数
    num_label = len(dataSet[0]) - 1  # 特征数量
    ent = entropy(dataSet)
    gain = 0.0  # 将最佳信息增益初始化为0
    label_axis = -1  # 最优特征的索引值
    # 遍历所有特征
    for i in range(num_label):
        label_list = [example[i] for example in dataSet]
        label_set = set(label_list)
        condition_ent = 0.0  # 初始化条件熵为0
        # 计算信息增益
        for label in label_set:
            set_after_split = splitDataSet(dataSet,i,label)
            Prob = len(set_after_split)/num_data
            condition_ent += Prob*entropy((set_after_split))
        gain1 = ent-condition_ent
        print("第%d个特征%s的增益为%.3f" % (i, labels[i], gain1))
        # 获得最佳信息增益
        if gain1 > gain:
            gain = gain1
            label_axis = i
    return label_axis

"""
创建决策树
"""
# ID3算法核心：以每个结点上的信息增益为选择的标准来递归的构建决策树
def createTree(dataSet,label):
    class_list = [example[-1] for example in dataSet]
    # 如果类别相同，则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        #count() 方法用于统计某个元素在列表中出现的次数。
        return class_list[0]
    bestFeat = split_(dataSet,label)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]    # 最优特征的标签
    mytree = {bestFeatLabel:{}}
    del(label[bestFeat])
    clasify_label_value = [example[bestFeat] for example in dataSet]
    set_clasify_label_value = set(clasify_label_value)
    # 遍历特征，创建决策树
    for value in set_clasify_label_value:
        new_label = label[:]
        # 构建数据的子集合，并进行递归
        mytree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),new_label)
    return mytree


if __name__=='__main__':
    dataSet,labels=creatDataSet()
    # print(dataSet,labels)
    mytree = createTree(dataSet,labels)
    print(f'决策树：{mytree}')
