import unittest

from dplearn.tree import *
from dplearn.datasets import insurement_v1
dataSet = [['长', '粗'],
            ['短', '粗'],
            ['短', '粗'],
            ['长', '细'],
            ['短', '细'],
            ['短', '粗'],
            ['长', '粗'],
            ['长', '粗']]

labels = ['男', '男','男', '女', '女', '女', '女', '女',]
#labels = ['头发','声音']  #两个特征

class Test_Criterion(unittest.TestCase):
    def test_criterion_entropy(self):
        """Criterion.entropy information
        
        -----
        PASS 0.9544340029249649
        """
        print(Criterion.entropy(*insurement_v1()))

class Test_Build_Tree(unittest.TestCase):
    def test_build_tree(self):
        """Tree.build_tree

        -----
        TODO
        """
        build_tree(*insurement_v1())

if __name__ == "__main__":
    unittest.main()
