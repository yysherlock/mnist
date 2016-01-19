import sys,os
import numpy as np
import unittest
import configparser

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def transform_output(data_label, length = 10):
    output = np.zeros([len(data_label), length]) # N x length
    targets = zip(np.array(range(data_label.shape[0])), data_label[:,0])
    for t in targets: output[t] = 1.0
    return output

def add_list(l1, l2):
    return [l1[i]+l2[i] for i in range(len(l1))]

def idxmapping(idx, rowsize, colsize):
    assert idx < rowsize*colsize, "idx error, out of index"
    row = idx / colsize
    col = idx % colsize
    return row,col

def euclid_norm(matrix):
    return np.sqrt(np.sum(matrix**2))

class NeuralNetwork(object):
    """
    Multiple layer neural NeuralNetwork
    """
    def __init__(self, architecture, opt):
        self.L = len(architecture) - 1
        self.architecture = architecture
        self.opt = opt
        # initial weights and biases
        self.weights = []
        self.biases = []
        for idx in xrange(len(architecture)-1):
            # follows N(0,1) distribution: np.random.randn(shape)
            # follows N(miu, sigma^2) distribution: sigma * np.random.randn(shape) + miu
            self.weights.append( 0.1 * np.random.randn( architecture[idx+1], architecture[idx] ) ) # L x H x V (append L times: V x H)
            self.biases.append( 0.1 * np.random.randn( architecture[idx+1], 1 ) ) # L x H (append L times: H x 1)
        self.learning_rate = opt['learning_rate']
        self.tolerance = opt['tolerance']
        self.batch_size = opt['batch_size']
        self.maxecho = opt['maxecho']

    def normalize(self, input_data):
        input_data = input_data / 255.0
        # normalzie to standard gaussian distribution
        return input_data

    def train(self, train_data, train_label):
        """stochastic gradient descent version of training"""
        converge = False
        iteration = 0
        start = 0
        org_label = train_label
        train_label = transform_output(train_label)

        while not converge and iteration < self.maxecho:

            #wgradients_magnitude = self.tolerance * np.ones(len(self.weights)) + 1
            #bgradients_magnitude = self.tolerance * np.ones(len(self.biases)) + 1
            if start == len(train_data): start = 0
            end = start + self.batch_size if start + self.batch_size <= len(train_data) else len(train_data)
            batch_data = train_data[start:end]
            batch_label = train_label[start:end]
            start = end

            wgradients, bgradients = self.mini_batch(batch_data, batch_label) # apply on all layers

            print 'cost:', self.cost(self.predict_output(train_data), train_label), \
            'acc:', self.evaluate(train_data, org_label)
            print self.weights
            print self.biases
            print '-------------------'

            wgradients_magnitude = np.array([ np.linalg.norm(wgradient) for wgradient in wgradients ])
            bgradients_magnitude = np.array([ np.linalg.norm(bgradient) for bgradient in bgradients ])
            self.update(wgradients, bgradients)
            if np.sum(wgradients_magnitude) < self.tolerance and np.sum(bgradients_magnitude) < self.tolerance:
                converge = True

            iteration += 1
            print 'iteration:',iteration

        print converge, np.sum(wgradients_magnitude), np.sum(bgradients_magnitude)

        print self.weights, self.biases

    def mini_batch(self, batch_data, batch_label):
        sz = batch_label.shape[0]
        avg_wgradients = [np.zeros(weight.shape) for weight in self.weights]
        avg_bgradients = [np.zeros(bias.shape) for bias in self.biases]

        for i in xrange(len(batch_label)):
            data = batch_data[i][np.newaxis].T # ith row, v x 1
            label = batch_label[i][np.newaxis].T #
            wgradients, bgradients = self.applygradient(data, label)
            avg_wgradients = add_list(avg_wgradients, wgradients)
            avg_bgradients = add_list(avg_bgradients, bgradients)

        for i in xrange(len(avg_wgradients)):
            avg_wgradients[i] = avg_wgradients[i]/float(sz)
        for i in xrange(len(avg_bgradients)):
            avg_bgradients[i] = avg_bgradients[i]/float(sz)

        return (avg_wgradients, avg_bgradients)

    def update(self, wgradients, bgradients):
        for idx in xrange(self.L):
            self.weights[idx] -= self.learning_rate * wgradients[idx]
            self.biases[idx] -= self.learning_rate * bgradients[idx]

    def computeNumericGradient(self, input, label, theta, epsilon=1e-4, sampleNum = 10):
        """ theta: w or b
        """
        h,v = theta.shape
        sample = np.random.randint(0, h*v, sampleNum)
        grad = np.zeros(sampleNum)

        for i,idx in enumerate(sample):
            # change theta
            theta[idxmapping(idx,h,v)] += epsilon
            c1 = self.getCost(input, label)
            theta[idxmapping(idx,h,v)] -= 2*epsilon
            c2 = self.getCost(input, label)
            grad[i] = (c1 - c2) / (2*epsilon)
            theta[idxmapping(idx,h,v)] += epsilon

        return grad, sample

    def applygradient(self, train_data, train_label):
        """ calculate wgradients, bgradients for all layers (L-1 ~ 0)
        d: represents dJ/dz at lth layer
        At current layer:
            a0 -- z1 -- a1
        """
        wgradients = []
        bgradients = []
        # generate outputs for each layer, i.e. ai for each layer i
        a0 = train_data # (784,1)
        outputs = [a0] # output[i] represents ai, i range from 0 to L
        for idx in xrange(self.L):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            a0 = self.feedforward(a0, weight, bias) # h x 1
            outputs.append(a0)

        # d: h x 1
        d = - outputs[self.L] * (1-outputs[self.L]) * (train_label-outputs[self.L]) # at layer L, initial, 10 x 1
        for l in xrange(self.L-1,-1,-1): # L-1 ~ 0
            # calculate gradient at layer l, l range from L-1 to 0
            # at layer l, current d is at l+1
            a0 = outputs[l]
            a1 = outputs[l+1]
            wgradient = np.dot(d,a0.T) # h x 1,1 x v -> h x v
            #print '==', 'w',l,':', wgradient, np.sum(wgradient != 0.0)
            bgradient = d # h x 1
            wgradients = [wgradient] + wgradients
            bgradients = [bgradient] + bgradients

            # gradient check
            numeric_wgradient, wsample = self.computeNumericGradient(train_data.T, train_label.T, self.weights[l])
            numeric_bgradient, bsample = self.computeNumericGradient(train_data.T, train_label.T, self.biases[l])
            for x,ws in enumerate(wsample):
                index = idxmapping(ws, *self.weights[l].shape)
                close = np.isclose(wgradient[index], numeric_wgradient[x])
                diffw = np.linalg.norm(wgradient[index] - numeric_wgradient[x])
                print 'diffw:', diffw, close

            for x,bs in enumerate(bsample):
                index = idxmapping(bs, *self.biases[l].shape)
                close = np.isclose(bgradient[index], numeric_bgradient[x])
                diffb = np.linalg.norm(bgradient[index] - numeric_bgradient[x])
                print 'diffb:', diffb, close
            #print 'diffw:', diffw, 'diffb:',diffb,", which should be very small"

            # update d for next layer, i.e. layer l-1
            d = np.dot(self.weights[l].T,d) * (a0*(1-a0)) # d at layer l
            #print '##',d
            #print '**',self.weights[l].T
            # v x h, h x 1 -> v x 1 * v x 1 -> v x 1
        return (wgradients, bgradients)

    def feedforward(self, a0, weight, bias):
        """ a0 -- z1 -- a1, return a1 """
        a1 = sigmoid(np.dot(weight, a0) + bias) # h x v, v x 1
        return a1 # h x 1

    def feedforward_all(self, a0, weight, bias):
        """ a0 -- z1 -- a1, return a1, for N examples """
        a1 = sigmoid(np.dot(weight, a0.T) + bias) # h x N
        return a1.T # N x h

    def predict(self,input_data): # N x v, N examples and V input/visual units
        output = self.predict_output(input_data) # N x 10
        return np.argmax(output, axis = 1).T #  1 x N

    def predict_output(self, input_data):
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            #input_data = sigmoid(np.dot(weight, input_data.T) + bias) # h x N
            #input_data = input_data.T # N x h, i.e. N x v in the next iteration
            input_data = self.feedforward_all(input_data, weight, bias) # N x h
        output = input_data # N x 10
        return output

    def get_output(self,input_data): # N x v, N examples and V input/visual units
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            input_data = self.feedforward(input_data, weight, bias) # 1 x h
        output = input_data # 1 x 10
        return output #  1 x N

    def cost(self, output, label):
        return 0.5 * np.sum((output - label)**2) #/ output.shape[0]

    def getCost(self, input_data, label): # for a mini batch
        output = self.predict_output(input_data)
        return self.cost(output, label)

    def evaluate(self, test_data, test_label):
        print 'predict shape:', self.predict(test_data).shape, 'label shape:',test_label.shape
        correct = np.sum(self.predict(test_data) == test_label.T) # test_label: 10000 x 1
        return correct, correct/float(len(test_data))

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

        opt = {'learning_rate':3.0, \
        'weight_decay': 1e-3, \
        'tolerance':0.01, \
        'batch_size':100, \
        'maxecho': 1000}
        self.nn = NeuralNetwork([784, 30, 10], opt)
        self.train_data = self.nn.normalize(self.train_data)
        self.test_data = self.nn.normalize(self.test_data)

    def test_NeuralNetwork(self):
        print self.train_label.shape, self.test_label.shape
        print self.train_data.shape, self.test_data.shape
        self.nn.train(self.train_data, self.train_label)
        print self.nn.evaluate(self.test_data, self.test_label)
        #print 'good'

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
