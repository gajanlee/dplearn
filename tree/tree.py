import math
class Tree:
    def __init__(self):
        pass


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
        