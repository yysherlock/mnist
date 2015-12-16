import unittest
import configparser
import matplotlib.pyplot as plt

from mnistProcessor import *

class TestMnistProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = mnistProcessor()
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        configfilePath = 'mnist-config.ini'
        self.config.read(configfilePath)

    def test_loadMNISTimages(self):
        train_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','train_images_file'))
        self.assertEqual(train_data.shape[1], 28)
        self.assertEqual(train_data.shape[2], 28)


    def test_loadLabels(self):
        train_label = self.processor.loadLabels(self.config.get('DEFAULT','train_labels_file'))
        self.assertEqual(train_label.shape[1], 1)

    def test_loadMNISTimages_show(self):
        train_data = self.processor.loadMNISTimages(self.config.get('DEFAULT','train_images_file'))
        im = train_data[0]
        plt.imshow(im, cmap='gray')
        plt.show()

if __name__=="__main__":
    unittest.main()
