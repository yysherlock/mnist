import sys,os
import numpy as np
import unittest
import configparser

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def transform_output(data_label, length = 10):
    output = np.zeros([len(data_label), length])
    targets = zip(np.array(range(data_label.shape[0])), data_label[:,0])
    for t in targets: output[t] = 1.0
    return output

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
            self.weights.append( 0.001 * np.random.randn( architecture[idx+1], architecture[idx] ) ) # L x H x V (append L times: V x H)
            self.biases.append( 0.001 * np.random.randn( architecture[idx+1], 1 ) ) # L x H (append L times: H x 1)
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

        while not converge and iteration < self.maxecho:

            #wgradients_magnitude = self.tolerance * np.ones(len(self.weights)) + 1
            #bgradients_magnitude = self.tolerance * np.ones(len(self.biases)) + 1
            if start == len(train_data): start = 0
            end = start + self.batch_size if start + self.batch_size <= len(train_data) else len(train_data)
            batch_data = train_data[start:end]
            batch_label = train_label[start:end]
            start = end

            wgradients, bgradients = self.applygradient(batch_data, batch_label) # apply on all layers

            print 'cost:', self.cost(self.predict_output(train_data), train_label)

            wgradients_magnitude = np.array([ np.linalg.norm(wgradient) for wgradient in wgradients ])
            bgradients_magnitude = np.array([ np.linalg.norm(bgradient) for bgradient in bgradients ])
            self.update(wgradients, bgradients)
            if np.all(wgradients_magnitude < self.tolerance) and np.all(bgradients_magnitude < self.tolerance):
                converge = True

            iteration += 1
            print 'iteration:',iteration

    def update(self, wgradients, bgradients):
        for idx in xrange(self.L):
            self.weights[idx] -= self.learning_rate * wgradients[idx]
            self.biases[idx] -= self.learning_rate * bgradients[idx]

    def applygradient(self, train_data, train_label):
        """ calculate wgradients, bgradients for all layers (L-1 ~ 0)
        d: represents dJ/dz at lth layer
        At current layer:
            a0 -- z1 -- a1
        """
        wgradients = []
        bgradients = []
        # generate outputs for each layer, i.e. ai for each layer i
        a0 = train_data # a0 = self.normalize(train_data)
        outputs = [a0] # output[i] represents ai, i range from 0 to L
        for idx in xrange(self.L):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            #print 'weight: ', weight.shape, 'bias:',bias.shape
            a0 = self.feedforward(a0, weight, bias) # N x h
            outputs.append(a0)

        labels = transform_output(train_label, 10) # N x 10

        # d: N x h
        #d = - outputs[self.L] * (1-outputs[self.L]) * np.sum(labels-outputs[self.L], axis=0)# at layer L, initial
        d = - outputs[self.L] * (1-outputs[self.L]) * (labels-outputs[self.L])
        for l in xrange(self.L-1,-1,-1): # L-1 ~ 0
            # calculate gradient at layer l, l range from L-1 to 0
            a0 = outputs[l]
            a1 = outputs[l+1]
            #print np.sum(d, axis = 0)[np.newaxis].T.shape,
            #wgradient = 1.0/len(train_data) * np.dot(a0 ,(np.sum(d, axis = 0)[np.newaxis].T * self.weights[l]).T)
            wgradient = 1.0/len(train_data) * np.dot(d.T,a0) # average gradient of all points
            bgradient = 1.0/len(train_data) * np.sum(d, axis = 0)[np.newaxis].T
            wgradients = [wgradient] + wgradients
            bgradients = [bgradient] + bgradients
            # update d for next layer, i.e. layer l-1
            #d = sigmoid(a0) * np.dot(d, self.weights[l]) # N x h, h x v -> N x v * N x v -> N x v
            d = a0*(1-a0) * np.dot(d, self.weights[l]) # N x h, h x v -> N x v * N x v -> N x v
            #d = 1.0/len(train_data) * np.dot(np.sum(d, axis = 0)[np.newaxis].T * self.weights[l], sigmoid(a0).T).T
        return (wgradients, bgradients)

    def computeNumericGradient(self, input, factor=1.0, eps=1e-4, sampleNum=500):
        """
        compute gradients throught numeric way for gradient check
        gradient of J w.r.t. x computed by (J(x+eps)-J(x-eps))/2eps
        only check param at sampleNum positions
        J=0.5*(a[0]-a[-1])**2+WeightCost
        """
        param=self.combineParam()
        plen=param.size
        if factor==0:
            plen=plen/2
        sample=np.random.randint(0,plen,sampleNum)
        grad=gp.zeros(sampleNum)
        for (i,idx) in enumerate(sample):
            if i%100==0:
                sys.stdout.write('.')
                sys.stdout.flush()
            q=gp.zeros(param.shape)
            q[idx]=eps
            p1=param+q
            p2=param-q
            c1,_=self.getCost(p1, input,factor)
            c2,_=self.getCost(p2, input,factor)
            grad[i]=(c1-c2)/(2.0*eps)
        print "end"
        return grad, sample

    def gradientCheck(self, dat):
        """
        check gradient by comparing with numeric computing
        it should be done on cpu
        """
        print "doing gradient check..."

        a=self.forward(dat)
        pgrad=self.computeGrads(a)
        pgrad=self.vectorParam(pgrad)

        pnumeric,idx=self.computeNumericGradient(dat)
        pgrad=pgrad[idx]
        diff= (pgrad-pnumeric).euclid_norm()/(pgrad+pnumeric).euclid_norm()
        print "the diff is %.15f, which should be very small" % diff


    def feedforward(self, a0, weight, bias):
        """ a0 -- z1 -- a1, return a1, for N examples """
        a1 = sigmoid(np.dot(weight, a0.T) + bias) # h x N
        return a1.T # N x h

    def predict(self,input_data): # N x v, N examples and V input/visual units
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            #input_data = sigmoid(np.dot(weight, input_data.T) + bias) # h x N
            #input_data = input_data.T # N x h, i.e. N x v in the next iteration
            input_data = self.feedforward(input_data, weight, bias) # N x h
        output = input_data # N x 10
        return np.argmax(output, axis = 1).T #  1 x N

    def predict_output(self, input_data):
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            #input_data = sigmoid(np.dot(weight, input_data.T) + bias) # h x N
            #input_data = input_data.T # N x h, i.e. N x v in the next iteration
            input_data = self.feedforward(input_data, weight, bias) # N x h
        output = input_data # N x 10
        return output

    def cost(self, output, label):
        return 0.5 * np.sum((output - label)**2)

    def evaluate(self, test_data, test_label):
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

        opt = {'learning_rate':1.0, \
        'weight_decay': 1e-3, \
        'tolerance':1e-3, \
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
