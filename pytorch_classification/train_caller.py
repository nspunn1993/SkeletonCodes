import argparse
from omegaconf import OmegaConf
from utils.train import ExperimentSetup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',  '-c',
                        dest="config",
                        metavar='Config file',
                        help =  'config file path',
                        default= 'config\cfg_train.yaml')
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Set the parameters
    train_params = cfg.train
    model_params = cfg.models
    vis_params = cfg.visualize
    data_params = cfg.data
    
    # Training
    exp = ExperimentSetup(train_params,model_params,vis_params,data_params.transform)

    exp.train()
    

    