import torchvision.transforms as transforms
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed

class Visualize:
    def __init__(self, dataset, labels, mode, idx_list=[], samples=10, cols=5, random_img = False, save_path=None):
        self.dataset = dataset
        self.labels = labels
        self.idx_list = idx_list
        self.samples = samples
        self.cols = cols
        self.random_img = random_img
        self.mode = mode
        self.save_path = save_path
        self.checkncreate_dir(save_path)
    
    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def visualize_samples(self, transform = 1):
        path = self.save_path
        dataset = copy.deepcopy(self.dataset)
        mode = self.mode
        org = dataset.transform_flag
        if transform == 0:
            dataset.transform_flag = False
            mode += '_notrans'
        else:
            dataset.transform_flag = True
        
        samples = min(self.samples,len(dataset))
        rows = samples // self.cols
        if rows == 0 or rows==1:
            cols = samples
            rows = 1
        else:
            cols = self.cols
        index_list = []
        process=[-1]
        idx=-1
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        for i in range(samples):
            if self.random_img and len(self.idx_list) > 0:
                idx = np.random.randint(0,len(self.idx_list))
            elif self.random_img and len(self.idx_list)==0:
                while(True):
                    if idx in process:
                        idx = np.random.randint(0,len(dataset))
                        continue
                    break
                process.append(idx)
            else:
                idx = self.idx_list[i]
            index_list.append(idx)
            image, lab = dataset[idx]
            image = np.array(image)
            if transform !=0:
                image = image.transpose([1, 2, 0])

            ax.ravel()[i].imshow(image[:,:,0], cmap='gray')
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(list(self.labels.keys())[int(lab.item(0))]+' - '+str(idx))
        plt.tight_layout(pad=1)
        plt.savefig(path+mode+'_samples.png',bbox_inches = "tight")
        plt.close()

        dataset.transform_flag = org
        
        return index_list
    
    def visualize_histograms(self, transform = 1):
        path = self.save_path
        dataset = copy.deepcopy(self.dataset)
        org = dataset.transform_flag
        mode = self.mode
        if transform == 0:
            dataset.transform_flag = False
            mode += '_notrans'
            xmax = 255.5
            width = 1
        else:
            dataset.transform_flag = True
            xmax = 5
            width = 1
    
        samples = min(self.samples,len(dataset))
        rows = samples // self.cols
        if rows == 0 or rows==1:
            cols = samples
            rows = 1
        else:
            cols = self.cols
        index_list = []
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 4))
        process=[-1]
        idx=-1
        for i in range(samples):
            if self.random_img and len(self.idx_list) > 0:
                idx = np.random.randint(0,len(self.idx_list))
            elif self.random_img and len(self.idx_list)==0:
                while(True):
                    if idx in process:
                        idx = np.random.randint(0,len(dataset))
                        continue
                    break
                process.append(idx)
            else:
                idx = self.idx_list[i]
            index_list.append(idx)
            image, lab = dataset[idx]
            image = np.array(image)
            ax.ravel()[i].hist(image.ravel(), bins=80)
            ax.ravel()[i].set_title(list(self.labels.keys())[int(lab.item(0))])
            if transform == 0:
                ax.ravel()[i].axis(xmin=-0.5,xmax=xmax,ymin=0,ymax=200000)
            else:
                ax.ravel()[i].axis(ymin=0,ymax=15000)
            
        plt.tight_layout(pad=1)
        plt.savefig(path+mode+'_hist_samples.png',bbox_inches = "tight")
        plt.close()
        
        dataset.transform_flag = org

        return index_list

    def visualize_predictions(self, y_pred, y_prob, idx_list = [], random_img = True):
        samples = min(10,len(self.dataset.indices))
        cols = 5
        rows = samples // cols
        if rows == 0 or rows==1:
            cols = samples
            rows = 1
            
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        index_list = []
        seed(32)
        process=[-1]
        idx=-1
        for i in range(samples):
            if random_img and len(idx_list) > 0:
                idx = np.random.randint(0,len(idx_list))
            elif random_img and len(idx_list)==0:
                while(True):
                    if idx in process:
                        idx = np.random.randint(0,len(self.dataset))
                        continue
                    break
                process.append(idx)
            else:
                idx = idx_list[i]
            index_list.append(idx)
            image, lab = self.dataset[int(self.dataset.indices[idx])]
            image = np.array(image)
            pred_lab = y_pred[idx]
            if int(lab) != int(pred_lab) and int(lab) != -1:
                color = 'r'
            else:
                color = 'k'
            pred_prob = round((y_prob[idx])*100,2)
            ax.ravel()[i].imshow(image[0], cmap='gray')
            ax.ravel()[i].set_axis_off()
            if lab != -1:
                ax.ravel()[i].set_title(str(idx)+'\nGT - '+list(self.labels.keys())[int(lab)][:] + ', Pred - '+list(self.labels.keys())[int(pred_lab)][:] + ',\nProb: '+str(pred_prob)+'%', color=color)
            else:
                ax.ravel()[i].set_title('Pred - '+list(self.labels.keys())[int(pred_lab)][:] + ',\nProb: '+str(pred_prob)+'%', color='k')
        plt.tight_layout(pad=1)
        plt.savefig(self.save_path+'predictions.png',bbox_inches = "tight")
        
        print('Saved Predictions at {}'.format(self.save_path+'predictions.png'))
        plt.close()

        return index_list