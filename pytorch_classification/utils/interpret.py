import matplotlib.pyplot as plt
import numpy as np
import torch
from random import seed
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class Interpret:
    
    def get_gradcams(dataset, model, y_true, y_pred, y_prob, idx_list = [], random_img = False, path='', tar_layer='', device='cuda:0'):  
        samples = min(10,len(dataset))
        labels = dataset._label_names
        cols = 5
        rows = samples // cols
        if rows == 0:
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
                        idx = np.random.randint(0,len(dataset))
                        continue
                    break
                process.append(idx)
            else:
                idx = idx_list[i]
            index_list.append(idx)
            image, lab = dataset[int(dataset.indices[idx])]
            #np.expand_dims(dataset[int(dataset.indices[idx])][0].numpy(),axis=0)
            image = np.expand_dims(np.array(image),axis=0)
            lab = y_true[idx]
            pred_lab = y_pred[idx]
            X = torch.tensor(image).float()
            y = [ClassifierOutputTarget(0)]
            
            X = X.to(device)

            image = np.float32(np.clip(image, 0, 1))
            image = np.transpose(image, [0,2,3,1])
            
            target_layers = [eval(tar_layer)]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            grayscale_cam = cam(input_tensor=X, targets=y)
            
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(np.squeeze(image), grayscale_cam, use_rgb=True)
            
            if int(lab) != int(pred_lab) and int(lab) != -1:
                color = 'r'
            else:
                color = 'k'
            pred_prob = round((y_prob[idx])*100,2)
            
            ax.ravel()[i].imshow(visualization)
            ax.ravel()[i].set_axis_off()
            if int(lab) != -1:
                ax.ravel()[i].set_title(str(idx)+'\nGT - '+list(labels.keys())[int(lab)][:] + ', Pred - '+list(labels.keys())[int(pred_lab)][:] + ',\nProb: '+str(pred_prob)+'%', color=color)
            else:
                ax.ravel()[i].set_title('Pred - '+list(labels.keys())[int(pred_lab)][:] + ',\nProb: '+str(pred_prob)+'%', color='k')
        plt.tight_layout(pad=1)
        plt.savefig(path+'gradcam.png',bbox_inches = "tight")
        print('Saved Gramdcams at {}'.format(path+'_Gradcam_.png'))
        plt.close()