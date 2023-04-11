import argparse
from omegaconf import OmegaConf
from utils.test import TestModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',  '-c',
                        dest="config",
                        metavar='Config file',
                        help =  'config file path',
                        default= 'config\cfg_test.yaml')
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Set the parameters
    test_params = cfg.test
    model_params = cfg.models
    vis_params = cfg.visualize
    data_params = cfg.data
    
    # Testing
    exp = TestModel(test_params,model_params,vis_params,data_params.transform)

    exp.test()
    

    