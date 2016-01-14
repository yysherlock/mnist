import sys,os
import unittest
import configparser

class NeuralNetwork(object):
    """
    Multiple layer neural NeuralNetwork
    """
    def __init__(self, architecture):
        self.n = len(architecture)
        self.architecture = architecture

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.processor = mnistProcessor()
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        configfilePath = '/home/luozhiyi/projects/mnist/config/mnist-config.ini'
        self.config.read(configfilePath)
        train_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','train_images_file'))
        train_label = self.processor.loadLabels(self.config.get('DEFAULT','train_labels_file'))
        test_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','test_images_file'))
        test_label = self.processor.loadLabels(self.config.get('DEFAULT','test_labels_file'))

    def test_NeuralNetwork(self):
        print 'good'
        #nn = NeuralNetwork([784, 30, 10])
        #nn.train(train_data, train_label)
        #print nn.evaluate(test_images_file, test_labels_file) # print classification accurracy

if __name__=="__main__":
    if __package__ is None: # not use as a package
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from util.mnistProcessor import *
    else: from ..util.mnistProcessor import * # use as a package
                                        # out of mnist dir: python -m mnist.NN.nn

    unittest.main()
