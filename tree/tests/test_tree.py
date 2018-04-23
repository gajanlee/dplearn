import unittest

from dplearn.tree import Criterion
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
        Criterion.entropy(dataSet, labels)


if __name__ == "__main__":
    unittest.main()
