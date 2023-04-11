import torchvision.transforms as transforms

class makeRGB(object):
    def __call__(self, img):
        return img.convert('RGB')


class DataTransformer:
    def __init__(self, **kwargs): #cfg_data = None, set_type='train', transformer=None
        height = kwargs['height']
        width = kwargs['width']
        set_type = kwargs['set_type']
        mean = kwargs['mean']
        std = kwargs['std']

        if set_type == 'train':
            
            transformer = transforms.Compose([
                                transforms.Resize((height, width)),
                                transforms.ToTensor()
                                ])
        
        elif set_type == 'valid' or set_type == 'test':

            transformer = transforms.Compose([
                                transforms.Resize((height, width)),
                                transforms.ToTensor(),
                                ])

        self.transformer = transformer