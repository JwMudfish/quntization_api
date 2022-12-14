from scipy.fft import dst
import torch
# from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import torch.nn as nn
import torchvision
import os
import time
import sys
# from utils import *
import timm
import glob
import copy
from torch.quantization import get_default_qconfig
# Note that this is temporary, we'll expose these functions to torch.quantization after official releasee
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import cv2
import warnings
import albumentations as A

warnings.filterwarnings(action='ignore')

# https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html
# fx qunti
# Use 'fbgemm' for server inference and 'qnnpack' for mobile inference.

class Mobilenet_v3(torch.nn.Module):
    def __init__(self, model_name='', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.model(x)
        return x

class Quntization:

    def __init__(self, model_path = '', device_type='', model_type='', network_name='', model_name='', version ='0'):
        self.model_path = model_path   #'./weight'
        self.device_type = device_type   #'fr'
        self.model_type = model_type     #'cls'
        self.network_name = network_name  #'mobilenetv3_small_075'  # mobilenetv3_small_075 고정
        self.model_name = model_name   #'patagonia_weisse_500_can_tspn'
        self.version =  version   #'0'

        self.image = self.read_image(img_path = './cali_img.jpg')
        self.weight_path = os.path.join(self.model_path, self.device_type, self.model_type, self.model_name, self.network_name, self.version)
        self.output_path = os.path.join(self.model_path, self.device_type, self.model_type, self.model_name, 'qt_mobilenetv3_small_075')


    def make_dir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f'{path} make folder')
        except OSError: 
            print("Error: Failed to create the directory.")

    def check_version(self, path : str):
        version_list = os.listdir(path)
        if len(version_list) == 0:
            new_version = '0'        
        else:
            check_version_folder = sorted(os.listdir(path), key = lambda x : int(x))
            new_version = str(int(check_version_folder[-1]) + 1)
        
        output_path = os.path.join(path, new_version)
        self.make_dir(output_path)
        
        return output_path

    def weight_selector(self, path):
        weights = glob.glob(path + '/*.pt') + glob.glob(path + '/*.pth')
        if len(weights) > 1:
            print(f'2 or more weight files! {weights[0]} is selected.')
        if len(weights) == 0:
            print(f'no weight file {path}!!')
        return weights[0]

    def load(self, weight_path, model):
        try:
            weight_file = self.weight_selector(self.weight_path)
            checkpoint = torch.load(weight_file)
            state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
            for key in list(state_dict.keys()):
                if 'module.' in key:
                    state_dict[key.replace('module.', '')] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict)
            model.to('cpu')
            model.eval()
        except Exception as e:
            print(e)
        return model

    def print_size_of_model(self, model):
        if isinstance(model, torch.jit.RecursiveScriptModule):
            torch.jit.save(model, "temp.p")
        else:
            torch.jit.save(torch.jit.script(model), "temp.p")
        print("Size (MB):", os.path.getsize("temp.p")/1e6)
        os.remove("temp.p")

    def calibrate(self, model, image):
        model.eval()
        with torch.no_grad():
            model(image)
    
    def read_image(self, img_path):
        image = np.array(cv2.imread(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, dsize=(224, 224),interpolation=cv2.INTER_LINEAR)
        trans = A.Compose([
        A.Resize(224, 224, p=1),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], p=1)
            ])
        image = trans(image = image)['image']
        image_swap = np.swapaxes(image, 0,2)
        image_swap = np.expand_dims(image_swap, axis=0)
        tensor = torch.from_numpy(image_swap).type(torch.FloatTensor)
        return tensor

    def load_base_model(self):
        base_model_path = os.path.join(self.model_path, self.device_type, self.model_type, self.model_name, self.network_name, self.version)
        base_model = Mobilenet_v3(model_name = self.network_name, pretrained=False)
        base_model = self.load(weight_path = self.model_path, model = base_model)

        base_model.eval()
        print('base model result : ', base_model(self.image))
        return base_model

    def run(self, verification = False):    

        base_model = self.load_base_model()
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}

        prepared_model = prepare_fx(base_model, qconfig_dict)  # fuse modules and insert observers

        # calibration
        self.calibrate(model = prepared_model, image = self.image)
        
        # convert the calibrated model to a quantized 
        quantized_model = convert_fx(prepared_model)  

        quantized_model.to('cpu')
        quantized_model.eval()
        print('quantized model result : ', quantized_model(self.image))
        
        self.make_dir(self.output_path)
        
        output_path = self.check_version(path = self.output_path)
        output_path = output_path + f'/qt_mobilenetv3_small_075_{self.model_name}_0fold_model.pth'
        
        torch.jit.save(torch.jit.script(quantized_model), output_path)
        
        if verification == True:
            #### 검증 ###

            self.print_size_of_model(base_model)
            self.print_size_of_model(quantized_model)

            q_model = torch.jit.load(output_path)
            # print(q_model['keys'])
            q_model.to('cpu')
            q_model.eval()

            print('!!! final result !!!!!')
            print(q_model(self.image))

        return quantized_model

if __name__ == '__main__':


    qt = Quntization(model_path = './weight',
                    device_type = 'fr',
                    model_type = 'cls',
                    network_name = 'mobilenetv3_small_075',  # mobilenetv3_small_075 고정
                    model_name = 'open_close_waterbath',
                    version = '0')
    qt.run(True)
