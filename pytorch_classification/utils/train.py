from utils.optimizers import Optimizer
from utils.loss import Loss
from utils.visualize import Visualize
from data.databuilder import BuildData
from models.models import BuildModel
from utils.eval import EvaluateModel
import torch
import numpy as np
import pandas as pd
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm.auto import tqdm
import pickle

class ExperimentSetup:
    def __init__(self, train_params, model_params, vis_params, trans_params):
        self.train_params = train_params # Training parameters from the config file
        self.model_params = model_params # Model parameters from the config file
        self.vis_params = vis_params # Visualization of training and validation samples
        self.trans_params = trans_params # Preprocessing parameters

    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    def save_model(self, model, mod_name):
        path = mod_name+"weights.pth"
        torch.save(model.state_dict(), path)
    
    def visualize_samples(self, dataset, data_type, path=''):
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

    def predict_proba(self, dataset, model, criterion, device, out_neurons):
        model.eval()
        pred_prob = []
        _lab = []
        total_loss = 0
        batches = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(dataset, 0):
                X , y = X.to(device), y.to(device)

                out = model(X)

                if out_neurons == 1:
                    loss = criterion(out, y.unsqueeze(1))
                    prob = torch.sigmoid(out)
                else:
                    loss = criterion(out, y.long())
                    prob = torch.softmax(out.data, 1)
                
                _lab.extend(y.detach().cpu().numpy().tolist())
                pred_prob.extend(prob.detach().cpu().numpy().tolist())

                total_loss += loss.item()
                batches += 1
        return _lab, pred_prob, total_loss/batches
    
    def save_pred_proba(self, dataset, path, predicted, label=None, epoch=0, type='train'):
        df = pd.DataFrame(columns=['FilePath','Proba', 'Label'])
        dict = {}
        for idx in dataset.indices:
            dict['FilePath']=[dataset._image_paths[idx]]
            dict['Proba']=[round((predicted[idx][0]),4)]
            dict['Label']=[label[idx]] if label is not None else ['']
            df = pd.concat([df, pd.DataFrame(dict)], ignore_index=True)
        self.checkncreate_dir(path+'epoch_csv//')
        df.to_csv(path+'epoch_csv//'+type+'_'+str(epoch)+'.csv')

    def train(self):
        # Training params
        epochs = self.train_params['epochs']
        device = self.train_params['device']
        batch_size = self.train_params['batch_size']
        patience = self.train_params['patience']
        save_path = self.train_params['save_path']
        optimizer = self.train_params['optimizer']
        lr = self.train_params['lr']
        scheduler = self.train_params['scheduler']
        lr_decay_factor = self.train_params['lr_decay_factor']
        weight_decay = self.train_params['weight_decay']
        loss = self.train_params['loss']
        tr_data_path = self.train_params['tr_path']
        vd_data_path = self.train_params['vd_path']

        # Visualization
        vis_samples = self.vis_params.vis
        vis_path = self.vis_params.path

        # Model params
        fr_layer=self.model_params['fr_layer']
        out_neurons=self.model_params['out_neurons']
        model_name=self.model_params['model_name']
        pretrained=self.model_params['pretrained']
        
        model_params = dict(fr_layer=fr_layer,
                            out_neurons=out_neurons,
                            model_name=model_name,
                            pretrained=pretrained)

        model_builder = BuildModel(**model_params)
        model = model_builder.get_model()
        model = model.to(device)
       
        # Data loader  batch_size=32, shuffle=True, path=[]
        train_data_builder = BuildData(self.trans_params, batch_size, shuffle=True, path=tr_data_path)
        valid_data_builder = BuildData(self.trans_params, batch_size, shuffle=False, path=vd_data_path)
        train_dataset = train_data_builder.build_data('train')
        valid_dataset = valid_data_builder.build_data('valid')
        train_loader = train_data_builder.build_loader()
        valid_loader = valid_data_builder.build_loader()

        class_weights = train_dataset.get_class_weights()
        class_weights = torch.tensor(class_weights).float().to(device)

        # Visualize Samples
        if vis_samples:
            self.visualize_samples(train_dataset, 'train', vis_path)
            self.visualize_samples(valid_dataset, 'valid', vis_path)
        
        # Directories
        inst_name = self.model_params['model_name']
        version = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.curr_version = version
        logs_dir = save_path + inst_name+"/" + version+"/"
        self.checkncreate_dir(logs_dir)
        writer = SummaryWriter(log_dir=logs_dir)
        print('Model version: {}'.format(version))

        # Optimizer and Scheduler
        optim = Optimizer(optim=optimizer,
                          lr=lr,
                          scheduler=scheduler,
                          patience=patience,
                          factor=lr_decay_factor,
                          wd=weight_decay,
                          params=model.parameters())

        optimizer = optim.get_optimizer()
        scheduler = optim.get_scheduler()

        # Loss function
        criterion = Loss(loss=loss, weights=class_weights).get_loss()

        # Resutls
        tr_results = []
        vd_results = []
        best_loss = 99999.0
        trigger = 0
        
        for epoch in tqdm(range(epochs)):
            model.train()
            loss_epoch = 0
            itr=0
            y_proba = []
            y_labels = []
            train_dataset.indices = []
            valid_dataset.indices = []
            
            for i, (X, y) in enumerate(train_loader, 0):
                X , y = X.to(device), y.to(device)

                optimizer.zero_grad()

                out = model(X)

                if out_neurons == 1:
                    loss = criterion(out, y.unsqueeze(1))
                    prob = torch.sigmoid(out)
                else:
                    loss = criterion(out, y.long())
                    prob = torch.softmax(out.data, 1)

                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                
                y_proba.extend(prob.detach().cpu().numpy())
                y_labels.extend(y.detach().cpu().numpy())
        
                itr += 1
            
            # Training Evaluation
            y_labels = np.squeeze(y_labels)
            y_proba = np.squeeze(y_proba)
            tr_loss = loss_epoch/itr

            tr_model_eval = EvaluateModel(y_labels, y_proba).scores
            tr_results.append(tr_model_eval)
            with open(logs_dir+'training_logs.pkl', 'wb') as fp:
                pickle.dump(tr_results, fp)
            
            # Validation Data Evaluation
            y_vd_labels, y_vd_proba, vd_loss = self.predict_proba(valid_loader, model, criterion, device, out_neurons)
            vd_model_eval = EvaluateModel(y_vd_labels, y_vd_proba).scores
            vd_results.append(vd_model_eval)
            with open(logs_dir+'validation_logs.pkl', 'wb') as fp:
                pickle.dump(vd_results, fp)
            
            # Summary Writer
            writer.add_scalar('Loss/Train',tr_loss,epoch+1)
            writer.add_scalar('Loss/Valid',vd_loss,epoch+1)
            tr_grid = vutils.make_grid(X)
            writer.add_image('Train input image', tr_grid, epoch+1)

            print('Epoch {}, train loss {}, valid loss {}'.format(epoch+1, tr_loss, vd_loss))
            
            if best_loss > vd_loss:
                self.save_model(model,logs_dir)
                self.save_pred_proba(train_dataset, logs_dir, y_proba, y_labels, epoch+1, type='train')
                self.save_pred_proba(valid_dataset, logs_dir, y_proba, epoch=epoch+1, type='valid')
                best_loss = vd_loss
                best_model = model
                trigger = 0
            else:
                trigger += 1
                print('Trigger count: {}'.format(trigger))

            if trigger > patience:
                print('Early stopping! '+inst_name+'_'+version)
                return best_model
            
            scheduler.step(vd_loss)

        print('Ending training for -- '+inst_name+'_'+version)
        
        return best_model
