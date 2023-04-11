from data.databuilder import BuildData
from utils.visualize import Visualize
from models.models import BuildModel
from utils.eval import EvaluateModel
from utils.interpret import Interpret
import torch
import pandas as pd
import os
import pickle

class TestModel:
    def __init__(self, test_params, model_params, vis_params, trans_params):
        self.test_params = test_params # Testing parameters from the config file
        self.model_params = model_params # Model parameters from the config file
        self.vis_params = vis_params # Visualization of samples
        self.trans_params = trans_params # Preprocessing parameters

    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def visualize_samples(self, dataset, data_type, path):
        vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
                                mode=data_type, idx_list=[], 
                                samples=10, cols=5, random_img = True, 
                                save_path=path)
        idx_list = vis_.visualize_samples(transform=0)
        vis_.idx_list = idx_list
        vis_.random_img = False
        vis_.visualize_samples(transform=1)
        vis_.visualize_histograms(transform=0)
        vis_.visualize_histograms(transform=1)
    
    def visualize_predictions(self, dataset, data_type, y_pred, y_prob, path):
        vis_ = Visualize(dataset=dataset, labels=dataset._label_names,
                                mode=data_type, idx_list=[], 
                                samples=10, cols=5, random_img = True, 
                                save_path=path)
        idx_list = vis_.visualize_predictions(y_pred, y_prob)
        return idx_list

    def predict_proba(self, dataset, model, device, out_neurons):
        model.eval()
        pred_prob = []
        pred = []
        _lab = []
        batches = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(dataset, 0):
                X , y = X.to(device), y.to(device)

                out = model(X)

                if out_neurons == 1:
                    prob = torch.sigmoid(out)
                else:
                    prob = torch.softmax(out.data, 1)
                
                _lab.extend(y.detach().cpu().numpy().tolist())
                pred_prob.extend(prob.detach().cpu().numpy().tolist())

                batches += 1
        
        return _lab, pred_prob

    def save_pred_proba(self, dataset, path, predicted, label=None):
        df = pd.DataFrame(columns=['FilePath','Proba', 'Label'])
        dict = {}
        for idx in dataset.indices:
            dict['FilePath']=[dataset._image_paths[idx]]
            dict['Proba']=[round((predicted[idx][0]),4)]
            dict['Label']=[label[idx]] if label is not None else ['']
            df = pd.concat([df, pd.DataFrame(dict)], ignore_index=True)
        df.to_csv(path+'test_results.csv')

    def test(self):
        # Testing params
        device = self.test_params['device']
        batch_size = self.test_params['batch_size']
        save_path = self.test_params['save_path']
        data_path = self.test_params['ts_path']
        interpret_layer = self.test_params['interpret_layer']

        # Model params
        out_neurons=self.model_params['out_neurons']
        model_name=self.model_params['model_name']
        model_weights = self.model_params['model_weights']

        # Visualization
        vis_samples = self.vis_params.vis
        vis_path = self.vis_params.path
        
        model_params = dict(out_neurons=out_neurons,
                            model_name=model_name)

        model_builder = BuildModel(**model_params)
        model = model_builder.get_model()

        # Load weights
        model.load_state_dict(torch.load(model_weights))
        model.to(device)

        # Data loader
        data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=data_path)
        test_dataset = data_builder.build_data('test')
        test_loader = data_builder.build_loader()

        # Visualize Samples
        if vis_samples:
            self.visualize_samples(test_dataset, 'test', vis_path)
            
        # Data Evaluation
        y_ts_labels, y_ts_proba = self.predict_proba(test_loader, model, device, out_neurons)
        
        ts_model_eval = EvaluateModel(y_ts_labels, y_ts_proba).scores
        index_list = self.visualize_predictions(test_dataset, 'test', ts_model_eval['y_pred'], ts_model_eval['y_prob'], save_path)

        self.save_pred_proba(test_dataset, save_path, y_ts_proba, y_ts_labels)
        with open(save_path+'test_results.pkl', 'wb') as fp:
            pickle.dump(ts_model_eval, fp)
        # dataset, model, y_true, y_pred, y_prob, idx_list = [], random_img = False, path='', tar_layer='', device='cuda:0'
        Interpret.get_gradcams(test_dataset, model, ts_model_eval['y'], ts_model_eval['y_pred'], ts_model_eval['y_prob'], index_list, False, save_path, interpret_layer, device)