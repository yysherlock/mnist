import configparser
import mnistProcessor
import NN

if __name__=="__main__":
    configfilePath="../config/mnist-config.ini"
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation()) # The interpolation parameter is important,
                                # this makes the `.ini` config file can use the `${}` syntax.
    config.read(configfilePath)
    #print config.keys() # config is sort of a dict
    #print config.get('DEFAULT','train_images_file')

    ## split data ##
    processor = mnistProcessor.mnistProcessor()
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    configfilePath = 'mnist-config.ini'
    config.read(configfilePath)

    train_data = processor.loadMNISTimages(config.get('DEFAULT','train_images_file'))
    train_label = processor.loadLabels(config.get('DEFAULT','train_labels_file'))
    test_data = processor.loadMNISTimages(config.get('DEFAULT','test_images_file'))
    test_label = processor.loadLabels(config.get('DEFAULT','test_labels_file'))


    nn = NN.NN([1, 2], configfilePath)
    nn.train(train_data, train_label)
