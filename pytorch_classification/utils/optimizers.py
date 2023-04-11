import torch
import torch_optimizer
from math import cos, pi

class Optimizer:
    def __init__(self, optim=None, lr=None, scheduler=None, patience=None, factor=None, params=None, iterations=None, wd=0):
        self.optim = optim
        self.lr = lr
        self.scheduler = scheduler
        self.patience = patience
        self.factor = factor
        self.params = params
        self.wd=wd
    
    def get_optimizer(self):
        if self.optim=='Adam':
            optimizer = torch.optim.Adam(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='AMSgrad':
            optimizer = torch.optim.Adam(self.params,lr=self.lr, amsgrad=True,weight_decay=self.wd)
        elif self.optim=='Adadelta':
            optimizer = torch.optim.Adadelta(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='RAdam':
            optimizer = torch_optimizer.RAdam(self.params,lr=self.lr,weight_decay=self.wd)
        elif self.optim=='SGD':
            optimizer = torch.optim.SGD(self.params,lr=self.lr,weight_decay=self.wd,momentum=0.9,nesterov=True)
        self.optimizer = optimizer
        return optimizer
    
    def get_scheduler(self):
        scheduler = ''
        if self.scheduler!='':
            if self.scheduler=='ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=self.factor)
            elif self.scheduler=='CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.01,mode='triangular2',cycle_momentum=False,
                                                step_size_up=350)
        
        
        return scheduler
