from NNLayer import *

class NNsoftmaxLayer(NNLayer):
    """
    softmax
    multi-classification: 0 - 9
    Layer Structure:
        (input) a0 --W,b--> z1 --sigmod--> a1 --softmax--> (output) yhat
        Here a0, z1, a1 are also 2D np arrays
        one row per case (example)
    """
    def __init__(self, vDim, hDim, a0):
        super(NNsoftmaxLayer, self).__init__(vDim, hDim, a0)

    def costFunction(self):
        pass

    def computeDlast(self):
        """
        dJ/dz1, this is the `d` for last last layer
        """
        

if __name__=="__main__":
    pass
