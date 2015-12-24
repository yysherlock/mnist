import numpy as np
#import gnumpy as gp

class NNLayer(object):
    """
    Layer Structure:
        (input) a0 --W,b--> z1 --sigmod--> a1
    """
    def __init__(self, hDim, vDim, a0):
        self.hDim = hDim
        self.vDim = vDim
        self.a0 = a0
        #init weight: uniform between +-sqrt(6)/sqrt(v+h+1)
        # gp.seed_rand() # Normal distribution, i.e. N(0,1)
        # r = gp.sqrt()
        r = np.sqrt(6) / np.sqrt(self.hDim + self.vDim + 1) # np -> gp
        # transfer this line to gnumpy: `self.W = gp.randn(self.vDim, self.hDim)`
        self.W = np.random.randn(self.vDim, self.hDim) * 2 * r - r
        self.b = np.random.randn(self.hDim, 1)

    def numericGradient(self, i, j, k, Jfunc):
        """ numerically checking the derivatives dJ/d\theta,
        here, \theta = {W,b}
        since, dJ/d{\theta} = lim_{\epsilon->0} (J(\theta + \epsilon) - J(\theta - \epsilon)) / (2*\epsilon)
        numerically approximate the derivative as follows:
        dJ/d\theta_i = (J(\theta_i + \epsilon) - J(\theta_i - \epsilon)) / (2 * \epsilon),
        where \epsilon is a very small number, say 10^{-4}

        utility `numpy.allclose OR numpy.isclose` methods
        """
        epsilon = 0.0001
        W1,b1,W2,b2 = self.W, self.b, self.W, self.b
        W1[i,j] = W1[i,j] + epsilon
        W2[i,j] = W2[i,j] - epsilon
        b1[k] = b1[k] + epsilon
        b2[k] = b2[k] - epsilon
        Wgradient = (Jfunc(W1, self.b) - Jfunc(W2, self.b)) / (2*epsilon)
        bgradient = (Jfunc(self.W, b1) - Jfunc(self.W, b2)) / (2*epsilon)
        print "--",Wgradient.shape, bgradient.shape,"--"
        return (Wgradient, bgradient) # dJ/dW_{ij}, dJ/db_k


    def Jfunc(W, b):
        pass

    def Jfunc_simple(self, W, b):
        """
        J(W,b) = W*b
        W_{vDim x hDim}, b_{hDim x 1}
        """
        return np.dot(W, b) # J(W,b) = W*b

    def computeGradient_simple(self):
        """
        compute the gradient for Jfunc_simple, J_{hDim x 1}
        J(W,b) = W*b + b
        """
        # J(j) = \sum_{i} W_{ij}*b_j + b_j, so dJ(j)/dW_{.j} = b_j, so Wgradient is vDim rows of [b_1, ..., b_j, ..., b_hDim]
        Wgradient = np.vstack([self.b] * self.vDim) # vDim x hDim
        # dJ/db_j = \sum_{i} Wij + 1
        # `nparray.sum(axis=0)` sum of rows, `nparray.sum(axis=1)` sum of cols
        bgradient = (self.W.sum(axis=0)).reshape(self.hDim,1)  # hDim x 1, hDim x 1
        return (Wgradient, bgradient)


    def checkGradient(self, points):
        flag = True
        for i,j,k in points:
            Wgradient, bgradient = self.computeGradient_simple()
            nWijgradient, nbkgradient = self.numericGradient(i, j, k, self.Jfunc_simple)
            if not np.allclose([Wgradient[i,j], bgradient[k,0]],[nWijgradient,nbkgradient]):
                print "computed gradients: ", [Wgradient[i,j], bgradient[k,0]]
                print "numeric gradients: ", [nWijgradient,nbkgradient]
                flag = False
                break
        return flag
