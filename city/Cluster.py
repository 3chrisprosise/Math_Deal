def readfile(filename):
    lines = [line for line in file(filename)]
    # 第一行是列标题
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # 每行的第一列是行名
        rownames.append(p[0])
        # 剩余部分就是该行的数据
        data.append([float(x) for x in p[1:］)
        return rownames, colnames, data
    # 计算皮尔逊相似值
    from math import sqrt
    def pearson(v1, v2):
        sum1 = sum(v1)
        sum2 = sum(v2)
        # 求平方和
        sum1Sq = sum([pow(v, 2) for v in v1])
        sum2Sq = sum([pow(v, 2) for v in v2])
        # 求乘积之和
        pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
        num = pSum - (sum1 * sum2 / len(v1))
        den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
        if den == 0: return 0
        return 1.0 - num / den

    class bicluster:
        def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
            self.left = left
            self.right = right
            self.vec = vec
            self.id = id
            self.distance = distance

    def hcluster(rows, distance=pearson):
        distances = {}
        currentclustid = -1;
        clust = [bicluster(rows[i], id=i) for i in range(len(rows))]
        while len(clust) > 1:
            lowestpair = (0, 1)
            closest = distance(clust[0].vec, clust[1].vec)
            for i in range(len(clust)):
                for j in range(i + 1, len(clust)):
                    if (clust[i].id, clust[j].id) not in distances:
                        distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                    d = distances[(clust[i].id, clust[j].id)]
                    if d < closest:
                        closest = d
                        lowestpair = (i, j)
            # 合并，计算两个聚类的平均值
            mergevec = [(clust[lowestpair[0].vec[i] + clust[lowestpair[1].vec[i])/2.0
            for i in range(len(clust[0].vec))]
            # 建立新的聚类
            newcluster = bicluster(mergevec, left = clust[lowestpair[0］, right = clust[lowestpair[1］,
            distance = closest, id = currentclustid)
            currentclustid -= 1
            del clust[lowestpair[1］
            del clust[lowestpair[0］
            clust.append(newcluster)
        return clust[0]

    def printClust(clust, labels=None, n=0):
        # 利用缩进建立分层布局
        for i in range(n): print ' '
        if clust.id < 0:
            # 负数表示这是一个分支
            print '-'
        else:
            # 正数表示这是一个节点
            if labels == None:
                print clust.id
            else:
                print labels[clust.id]
        # 递归打印
        if clust.left != None: printClust(clust.left, labels=labels, n=n + 1)
        if clust.right != None: printClust(clust.right, labels=labels, n=n + 1)

    from PIL import Image, ImageDraw
    # 计算树的高度
    def getheight(clust):
        if clust.left == None and clust.right == None: return 1
        return getheight(clust.left) + getheight(clust.right)

    # 计算根节点深度
    def getdepth(clust):
        if clust.left == None and clust.right == None: return 0
        return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance

    # 绘制树状图
    def drawdendrogram(clust, labels, jpeg='clusters.jpg'):
        # 高度和宽度
        h = getheight(clust) * 20
        w = 1200
        depth = getdepth(clust)
        # 计算缩放因子，调整距离值
        scaling = float(w - 150) / depth
        # 新建一个白色背景的图片
        img = Image.new('RGB', (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))
        # 绘制第一个节点
        drawnode(draw, clust, 10, (h / 2), scaling, labels)
        img.save(jpeg, 'JPEG')

    def drawnode(draw, clust, x, y, scaling, labels):
        if clust.id < 0:
            h1 = getheight(clust.left) * 20
            h2 = getheight(clust.right) * 20
            top = y - (h1 + h2) / 2
            bottom = y + (h1 + h2) / 2
            # 线的长度
            ll = clust.distance * scaling
            # 聚类到其子节点的垂直线
            draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
            # 连接左侧节点的水平线
            draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))
            # 连接右侧节点的水平线
            draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))
            ##递归绘制左右节点
            drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, labels)
            drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, labels)
        else:
            # 如果是叶节点，则绘制标签
            draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))

    # 按列聚类
    def rotateMatrix(data):
        newdata = []
        for i in range(len(data[0])):
            newrow = [data[j][i] for j in range(len(data))]
            newdata.append(newrow)
        return newdata

    # k-means聚类
    import random
    def kcluster(rows, distance=pearson, k=4):
        # 确定每个点的最小值和最大值
        ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
                  for i in range(len(rows[0]))]
        # 随机创建k个中心点
        clusters = [random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        for i in range(len(rows[0])):
            for j in range(k):
                lastmatches = None
        for t in range(100):
            print('Iteration %d' % t)
        bestmatches = []for i in range(k)]
        # 在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row = rows[j]
        bestmatch = 0
        for i in range(k):
            d = distance(clusters[i], row)
            if d < distance(clusters[bestmatch], row): bestmatch = i
                bestmatches[bestmatch].append(j)
            if bestmatches == lastmatches:
                break

            lastmatches = bestmatches
        # 将中心点移到所有成员的中心处
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs

    return bestmatches



# -*- coding: cp936 -*-
class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


# 在某一列上对数据集进行拆分
def divideset(rows, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def uniquecount(rows):
    results = {}
    for row in rows:
        # 计数结果在最后一列
        r = row[len(row) - 1]
        # 初始化
        if r not in results: results[r] = 0
        results[r] += 1
    return results


# 随机放置的数据项出现在错误分类的概率
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecount(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = uniquecount(rows)
    ent = 0.0
    for r in results:
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def buildtree(rows, scoref=entropy):
    if len(rows) == 0: return decisionnode()
    current_score = scoref(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]]=1
            # 根据这一列中的每个值，尝试对数据集进行拆分
            for value in column_values.keys():
                (set1, set2) = divideset(rows, col, value)
            # 信息增益
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
            best_criteria = (col, value)
            best_sets = (set1, set2)
            if best_gain > 0:
                trueBranch = buildtree(best_sets[0])
            falseBranch = buildtree(best_sets[1])
            return decisionnode(col=best_criteria[0], value=best_criteria[1],
                                tb=trueBranch, fb=falseBranch)
        else:
            return decisionnode(results=uniquecount(rows))


def printtree(tree, indent=' '):
    if tree.results != None:
        print(str(tree.results))
    else:
        print(str(tree.col) + ':' + str(tree.value) + '? ')
        print(indent + 'T->')
        printtree(tree.tb, indent + ' ')
        print(indent + 'F->')
        printtree(tree.fb, indent + ' ')


# 图形化方式
def getwidth(tree):
    if tree.tb == None and tree.fb == None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


from PIL import Image, ImageDraw


def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.results == None:
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


# 分类预测函数
def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
    return classify(observation, branch)


# 剪枝
def prune(tree, mingain):
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)
    if tree.tb.results != None and tree.fb.results != None:
        tb, fb = [], []
        for v, c in tree.fb.results.items():
            tb += [v]*c
        for v, c in tree.fb.results.items():
            fb += [v]*c
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = uniquecount(tb + fb)


def mdclassify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        if v == None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items(): result[k] = v * tw
            for k, v in fr.items():
                if k not in result: result[k] = 0
                result[k] += v * fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


# 计算方差
def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance