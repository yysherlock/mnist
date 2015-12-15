import os, struct
import numpy as np
#import gnumpy as gp
import matplotlib.pyplot as plt

class mnistProcessor(object):
    """ processing functionality for mnist images """

    def __init__(self):
        super(mnistProcessor,self).__init__()

    def loadMNISTimages(self, filename):
        """ mnist images files are binary files.
        Read them into numpy(or gnumpy) arrays
        """
        binf = open(filename, 'rb')
        # `>IIII` means read 4 unsigned int32 as `>` code way, i.e. 4*32 bits = 16 Bytes,
        # so the second param is `binf.read(16)`
        magic_nr, size, rows, cols = struct.unpack(">IIII",binf.read(16))

        # We use `numpy.uint8` 2D array to store image pixels,
        # since intensity of image pixels range from 0 to 255 (2^8 = 256).
        data = np.empty([size, rows, cols], dtype=np.uint8) # so one pixel is one byte
        perimbytes = rows*cols
        for i in range(size):
            im = np.array(struct.unpack(">"+str(perimbytes)+"B", binf.read(perimbytes))).reshape(rows, cols)
            data[i] = im
        binf.close()

        """
        #im = struct.unpack(">784B",binf.read(28*28))
        #im = np.array(im).reshape(28,28)
        im = data[0]
        # print im.shape
        plt.imshow(im, cmap='gray')
        plt.show()
        """
        # print data.shape
        return data # size x rows x cols numpy array

    def loadLabels(self, filename):
        binf = open(filename, 'rb')
        magic_nr, size = struct.unpack(">II", binf.read(8))
        data = np.empty([size,1], dtype = np.int8) # label range from 0 to 9
        for i in range(size):
            data[i] = np.array(struct.unpack(">1B", binf.read(1)))
        binf.close()
        # print type(data[0]), data[0].shape, data[0]
        return data # size x 1 numpy array
