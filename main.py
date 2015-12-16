import configparser
import mnistProcessor
import NN

def train(configfilePath):
    pass

if __name__=="__main__":
    configfilePath="mnist-config.ini"
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation()) # The interpolation parameter is important,
                                # this makes the `.ini` config file can use the `${}` syntax.
    config.read(configfilePath)
    #print config.keys() # config is sort of a dict
    #print config.get('DEFAULT','train_images_file')
    """
    processor = mnistProcessor.mnistProcessor()
    train_data = processor.loadMNISTimages(config.get('DEFAULT','train_images_file'))
    train_label = processor.loadLabels(config.get('DEFAULT','train_labels_file'))
    test_data = processor.loadMNISTimages(config.get('DEFAULT','test_images_file'))
    test_label = processor.loadLabels(config.get('DEFAULT','test_labels_file'))
    """
    nn = NN.NN([1,2])
    nn.computeDlast()
