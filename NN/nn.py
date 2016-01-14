import sys,os
import numpy as np
import unittest
import configparser

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class NeuralNetwork(object):
    """
    Multiple layer neural NeuralNetwork
    """
    def __init__(self, architecture, opt):
        self.n = len(architecture)
        self.architecture = architecture
        # initial weights and biases
        self.weights = []
        self.biases = []
        for idx in xrange(len(architecture)-1):
            # follows N(0,1) distribution: np.random.randn(shape)
            # follows N(miu, sigma^2) distribution: sigma * np.random.randn(shape) + miu
            self.weights.append( np.random.randn( architecture[idx], architecture[idx+1] ) ) # L x V x H (append L times: V x H)
            self.biases.append( np.random.randn( architecture[idx+1], 1 ) ) # L x H (append L times: H x 1)
        self.learning_rate = opt['learning_rate']
        self.tolerance = opt['tolerance']
        self.batch_size = opt['batch_size']
        self.echo = opt['maxecho']

    def train(self, train_data, train_label):
        """stochastic gradient descent version of training"""
        converge = False
        iteration = 0

        while not converge and iteration < self.opt.maxecho:
            #wgradients_magnitude = self.tolerance * np.ones(len(self.weights)) + 1
            #bgradients_magnitude = self.tolerance * np.ones(len(self.biases)) + 1
            wgradients, bgradients = self.applygradient() # apply on all layers
            wgradients_magnitude = np.array([ np.linalg.norm(wgradient) for wgradient in wgradients ])
            bgradients_magnitude = np.arrya([ np.linalg.norm(bgradient) for bgradient in bgradients ])

            if np.all(wgradients_magnitude < self.tolerance) and np.all(bgradients_magnitude < self.tolerance):
                converge = True

            iteration += 1

    def predict(self,input_data): # N x v, N examples and V input/visual units
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight.T: h x v
            bias = self.biases[idx] # bias: h x 1
            input_data = sigmoid(np.dot(weight.T, input_data.T) + bias) # h x N
            input_data = input_data.T # N x h, i.e. N x v in the next iteration
        output = input_data # N x 10
        return np.argmax(output, axis = 1).T #  1 x N

    def evaluate(self, test_data, test_label):
        correct = np.sum(self.predict(test_data) == test_label.T) # test_label: 10000 x 1
        return correct/float(len(test_data))

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.processor = mnistProcessor()
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        configfilePath = '/home/luozhiyi/projects/mnist/config/mnist-config.ini'
        self.config.read(configfilePath)
        self.train_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','train_images_file'))
        self.train_label = self.processor.loadLabels(self.config.get('DEFAULT','train_labels_file'))
        self.test_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','test_images_file'))
        self.test_label = self.processor.loadLabels(self.config.get('DEFAULT','test_labels_file'))

        opt = {'learning_rate':0.001, \
        'tolerance':1e-3, \
        'batch_size':100, \
        'maxecho': 5000}
        self.nn = NeuralNetwork([784, 30, 10], opt)

    def test_NeuralNetwork(self):
        print self.train_label.shape, self.test_label.shape
        print self.train_data.shape, self.test_data.shape
        print self.nn.evaluate(self.test_data, self.test_label)
        print 'good'

        #nn = NeuralNetwork([784, 30, 10], opt)
        #nn.train(train_data, train_label)
        #print nn.evaluate(test_images_file, test_labels_file) # print classification accurracy

if __name__=="__main__":
    if __package__ is None: # not use as a package
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from util.mnistProcessor import *
    else: from ..util.mnistProcessor import * # use as a package
                                        # out of mnist dir: python -m mnist.NN.nn

    unittest.main()
