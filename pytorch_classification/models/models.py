import torchvision.models as models
import torch.nn as nn

class BuildModel:
    def __init__(self, **kwargs):
        self.fr_layer = kwargs['fr_layer'] if 'fr_layer' in kwargs else 0
        self.out_neurons = kwargs['out_neurons'] if 'out_neurons' in kwargs else 1
        self.model_type = kwargs['model_name'] if 'model_name' in kwargs else 'resnet18'
        self.pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else False

    # Count number of layers in a model
    def len_layers(self, model):
        count = 0
        for param in model.parameters():
            count += 1
        return count

    # Count number of training parameters
    def count_tr_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count number of non-trainable parameters
    def count_ntr_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    def print_model_layers_names(self, model):
        layer_num = 1
        for name, layer in model.named_modules():
            print(layer_num, name)
    
    # Get number of last layers to freeze
    def num_freeze_layers(self, model, block):
        count = 0
        flg = 0
        for layer_name, param in reversed(list(model.named_parameters())):

            if flg == 1:
                if block in layer_name:
                    count += 1
            else:
                count += 1
                if block in layer_name:
                    flg = 1
        return count

    def get_model(self):
        lay = self.fr_layer # Number of last layers to enabler training
        if self.model_type.lower() == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained, zero_init_residual=True)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained, zero_init_residual=True)
            filters = model.fc.in_features
            model.fc = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'densenet161':
            model = models.densenet161(pretrained=self.pretrained, drop_rate = 0.3)
            filters = model.classifier.in_features
            model.classifier = nn.Linear(filters, self.out_neurons)

        elif self.model_type.lower() == 'densenet121':
            model = models.densenet121(pretrained=True)
            filters = model.classifier.in_features
            model.classifier = nn.Linear(filters, self.out_neurons)
            
        else:
            return 'Model option not available'
        
        if not self.pretrained:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
        if isinstance(lay, str):
            lay = self.num_freeze_layers(model, lay)

        if lay != 0:
            layers = self.len_layers(model)
            print(layers)
            for param in model.parameters():
                if layers == lay:
                    break
                param.requires_grad = False
                layers -= 1

        #model = nn.DataParallel(model)

        print('Model {}.........'.format(self.model_type.lower()))
        print('Number of training parameters: {}'.format(str(self.count_tr_parameters(model))))
        print('Number of non training parameters: {}'.format(str(self.count_ntr_parameters(model))))
        
        return model