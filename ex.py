def itemset(dataset):
    a=[]
    for i in dataset:
        for j in i:
            if [j] not in a:
                a.append(j)
    return a

def freq_itemset(dataset,a,min_freq):
    cut_branch={}

    for i in a:
        for j in dataset:
            if set(i).issubset(set(j)):
                # 判断集合的所有元素是否都包含在指定集合中
                cut_branch[tuple(i)]=cut_branch.get(tuple(i),0)+1
                # 通过tuple函数将字典转化为元组，保留键值，通过get获取其键的值
                # 剪枝法，将元素加入字典中并统计出现的次数。
    Lk=[]
    L1={}

    for i in cut_branch:
        if cut_branch[i]>=min_freq: # 当支持度大于最小阈值：
            Lk.append(list(i))
            # 频繁项集
            L1[i]=cut_branch[i]
            # L1用于存储存放所有频繁项集的支持度

    return L1,Lk

def KtoKplus(Lk,K):
    candidate=[]
    # 存放候选集
    for i in range(len(Lk)):
        for j in range(i+1,len(Lk)):
            L1=list(Lk[i])[:K-2]
            L2=list(Lk[j])[:K-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                if K > 2:
                    new=list(set(Lk[i]) ^ set(Lk[j]))
                else:
                    new=set()
                for m in Lk:
                    if set(new).issubset(set(m)) and list(set(Lk[i])) | set(Lk[j]) not in candidate:
                        candidate.append( list(set(Lk[i]) | set(Lk[j])) )
        return candidate

def Apriori( dataset , min_freq = 2 ):
    a = itemset(dataset)
    f1, sup_1 = freq_itemset(dataset, a, min_freq)
    F=[f1]
    sup_data=sup_1
    K=2
    while(len(F[K-2])>1):
        candidate=KtoKplus(F[K-2],K)
        fk,sup_k=freq_itemset(dataset,candidate,min_freq)
        F.append(fk)
        sup_data.update(sup_k)
        K+=1
    return F,sup_data

if __name__=='__main__':
    dataset=[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    F, sup_data = Apriori(dataset, min_freq = 2)

    print(">>with corresponds are:{} ".format(F))
    print("corresponing support is:{}".format(sup_data))