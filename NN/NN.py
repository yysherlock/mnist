import numpy as np

class NN(object):
    """

    """
    def __init__(self, listOflayers, computeDlast = None):
        self.layers = [] # NN: a0 --W1.b1--> z1 --f--> a1 --W2,b2--> z2 --f--> a2
        if computeDlast: # we can assign this function for different cost function J
            self.computeDlast = computeDlast
        if len(layers) > 0:
            for i in range(len(listOflayers)):
                vdim = listOflayers[i]
                hdim = listOflayers[i+1]

    def init(self):
        pass

    def getActivationGradient(self, a):
        """ a=f(z), da/dz = a(1-a), f is sigmod function """
        return a*(1-a)

    def computeBackwardD(self, d2, W, a1):
        """ calculate dJ/dz^{l} (e.g. d1), needs dJ/dz^{l+1} (e.g. d2), """
        d1 = d2 * getActivationGradient(a1)
        return d1

    def computeDlast(self, z, y, W): # default J is residual square sum error 0.5||y-x||^2
        # print "hello world"
        """ calculate dJ/dz, where z is output of last layer"""

        return (y-z)*self.getActivationGradient(a)

    
