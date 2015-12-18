import unittest

from NNLayer import *

class TestNNLayer(unittest.TestCase):

    def setUp(self):
        self.nnlayer = NNLayer(3,2) # vDim = 3, hDim = 2

    def test_checkGradient(self):
        self.assertEqual(self.nnlayer.checkGradient([(0,0,0),(0,0,1),(0,1,1),(2,1,1)]), True)

if __name__=="__main__":
    unittest.main()
