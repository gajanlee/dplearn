import math
class Tree:
    def __init__(self):
        self.children = {}  # the key is criterion attribute, the value is subtree node.
    
    def build_tree(self, dataX, y):
        pass
    
    def set_attr(self, attr_name):
        """Set current Node's attribute, such as: 有房
        """
        self.attr_name = attr_name
        return self
    
    def set_res(self, res_name):
        """Set current Node's classified name, in the leaf node, such as: 是/否
        """
        self.res_name = res_name
        return self
    
    def __str__(self):
        for c, n in self.children.items():
            print()
    

def build_tree(dataX, labels, attns=[]):
    node = Tree()
    classes = list(set(labels))
    if len(classes) == 1:
        node.set_res(labels[0])
        return node
    # TODO: A is None

    sum_entropy = Criterion.entropy(dataX, labels)

    # divide dataset to compute respective infomation entropy 
    entropies = [0]*len(dataX[0])
    counts = []
    for i in range(len(entropies)):
        count = {}
        for cate, _y in zip(dataX, labels): # _y代表训练集的分类结果
            count[cate[i]] = count.get(cate[i], []) + [_y]  # 为每一种属性的结果集赋值，如{'青年': ['否', '否', '是', '是', '否'], '中年': ['否', '否', '是', '是', '是'], '老年': ['是', '是', '是', '是', '否']}
        entropies[i] = sum_entropy - sum([(len(res) / len(labels)) * Criterion.entropy([attn]*len(res), res) for attn, res in count.items()]) # 计算第n项的信息增益
        counts.append(count)
    
    # 找到信息增益最大的分类标准, 记录第i项最大
    ms = max([(e, i) for i, e in enumerate(entropies)])[1]
    
    _split = {c: ([], []) for c in counts[ms]}
    for i, dx in enumerate(dataX):
        _split[dx[ms]][0].append(dx[:ms] + dx[ms+1:])        
        _split[dx[ms]][1].append(labels[i])
    
    print(_split)
    for sp, (_X, _y) in _split.items():
        node.children[sp] = build_tree(_X, _y)
    print(node.children)    
    return node
    



class Criterion:
    """A criterion rule for nodes selection.

    """
    def __init__(self):
        pass
    
    @staticmethod
    def gini():
        pass
    
    @staticmethod
    def entropy(dataset, labels):
        """information entropy. 
        """
        assert len(dataset) == len(labels)
        classes = list(set(labels))
        if len(classes) == 1: return  0
        record = [0]*len(classes)
        for i, data in enumerate(dataset):
            record[classes.index(labels[i])] += 1

        return -sum([r/len(dataset)*math.log(r/len(dataset), len(classes)) for r in record]) 
        

    def funcname(self, parameter_list):
        raise NotImplementedError

class Splitter:
    """
    Splitter strategy, default is "best". Also, random fortress.
    """
    pass


"""
Decision Tree
"""

dataX = [
    ["青绿", "蜷缩", "浊响"],
    ["乌黑", "蜷缩", "沉闷"],
]
dataY = [
    1, 1, 1, 1, 1

]

labels = ["色泽", "根茎"]



class DecisionTree:
    def __init__(self):
        pass
    
    def fit(self, X, y, labels):
        cates = {}
        for i, _y in enumerate(y): cates[_y] = cates.get(_y, []).append(i)

        node = Tree()
        # 如果所有的分类都相同， 树节点设置为该类别
        if len(set(y)) == 1:
            node.result = y[0]; return node
        
        
        _classes = list(set(y))
        result = Criterion.entropy(X, _classes)
        dx = {}
        for attn in range(len(labels)):
            for c in range(len(_y)):
                for i, x in enumerate(X):
                    dx[x[attn]] = dx.get(x[attn], [0]*len(_y))[_y.index(y[i])] + 1
            _y = set(y)
            for c in range(len(_y)):
                dx = [[x[attn] for i, x in enumerate(X) if dataY[i] == _y[c]] for c in range(len(set(y)))]  # [[青绿， 青绿]]
            Gain(dx)

        if sum(y) == len(y) or sum(y) == 0:
            node.result = y[0]; return node
        