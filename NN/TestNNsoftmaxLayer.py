import unittest

from NNsoftmaxLayer import *

class TestNNsoftmaxLayer(unittest.TestCase):

    def setUp(self):
        self.softmaxlayer = NNsoftmaxLayer(3,2)

    def test_dirlayer(self):
        """ make sure that softmaxlayer contains `W` and `b` parameters """
        print dir(self.softmaxlayer)
        self.assertEqual('W' in dir(self.softmaxlayer),True)

if __name__=="__main__":
    unittest.main()
