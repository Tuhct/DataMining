def item(dataset):  # 求第一次扫描数据库后的 候选集，（它没法加入循环）
    a = []  # 存放候选集元素

    for i in dataset:  # 就是求这个数据库中出现了几个元素，然后返回
        for j in i:
            if [j] not in a:
                a.append([j])
    return a


def get_frequent_item(dataset, a, min_support):
    cut_branch = {}  #设置字典存放各项集及其支持度
    for i in a:
        for j in dataset:
            if set(i).issubset(set(j)):
                cut_branch[tuple(i)] = cut_branch.get(tuple(i),
                                                      0) + 1  # cut_branch[y] = new_cand.get(y, 0)表示如果字典里面没有想要的关键词，就返回0
    # print(cut_branch)

    Lk = []  # 支持度大于最小支持度的项集，  即频繁项集
    L1 = {}  # 用来存放所有 频繁 项集的支持度的字典

    for i in cut_branch:
        if cut_branch[i] >= min_support:  # Apriori定律1  小于支持度，则就将它舍去，它的超集必然不是频繁项集
            Lk.append(list(i))
            L1[i] = cut_branch[i]
    # print(Fk)
    return Lk, L1


def get_candidate(Lk, K):  # 求第k次候选集
    candidate = []  # 存放产生候选集

    for i in range(len(Lk)):
        for j in range(i + 1, len(Lk)):
            L1 = list(Lk[i])[:K - 2]
            L2 = list(Lk[j])[:K - 2]
            L1.sort()
            L2.sort()  # 先排序，在进行组合

            if L1 == L2:
                if K > 2:  # 第二次求候选集，不需要进行减枝，因为第一次候选集都是单元素，且已经减枝了，组合为双元素肯定不会出现不满足支持度的元素
                    new = list(set(Lk[i]) ^ set(Lk[j]))  # 集合运算 对称差集 ^ （含义，集合的元素在t或s中，但不会同时出现在二者中）
                    # new表示，这两个记录中，不同的元素集合
                    # 为什么要用new？ 比如 1，2     1，3  两个合并成 1，2，3   我们知道1，2 和 1，3 一定是频繁项集，但 2，3呢，我们要判断2，3是否为频繁项集
                    # Apriori定律1 如果一个集合不是频繁项集，则它的所有超集都不是频繁项集
                else:
                    new = set()
                for x in Lk:
                    if set(new).issubset(set(x)) and list(
                            set(Lk[i]) | set(Lk[j])) not in candidate:  # 减枝 new是 x 的子集，并且 还没有加入 ck 中
                        candidate.append(list(set(Lk[i]) | set(Lk[j])))
    # print(ck)
    return candidate


def Apriori(dataset, min_support=2):
    c1 = item(dataset)  # 返回一个二维列表，里面的每一个一维列表，都是第一次候选集的元素
    f1, sup_1 = get_frequent_item(dataset, c1, min_support)  # 求第一次候选集

    F = [f1]  # 将第一次候选集产生的频繁项集放入 F ,以后每次扫描产生的所有频繁项集都放入里面
    sup_data = sup_1  # 一个字典，里面存放所有产生的候选集，及其支持度

    K = 2  # 从第二个开始循环求解，先求候选集，在求频繁项集

    while(len(F[K - 2]) > 1):  # k-2是因为F是从0开始数的     #前一个的频繁项集个数在2个或2个以上，才继续循环，否则退出
        candidate = get_candidate(F[K - 2], K)  # 求第k次候选集
        fk, sup_k = get_frequent_item(dataset, candidate, min_support)  # 求第k次频繁项集

        F.append(fk)  # 把新产生的候选集假如F
        sup_data.update(sup_k)  # 字典更新，加入新得出的数据
        K += 1
    return F, sup_data  # 返回所有频繁项集， 以及存放频繁项集支持度的字典


if __name__ == '__main__':
    dataset = [['a', 'b', 'c'], ['a', 'c', 'd'], ['a', 'b', 'c', 'd'], ['b', 'e']]
    F, sup_data = Apriori(dataset, min_support=2)

    print(">>with corresponds{}".format(F))

    print(">>support{}".format(sup_data))


