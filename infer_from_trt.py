# convert pt model int ts 
# do the inference -- use it for orin, (to debug)
import os 
import yaml
import pprint 

import numpy as np 
import torch
import torch_tensorrt

from utils import * 
from create_model import *



def build_tensorrt_model(cfg):
    device = 'cuda' if cfg['tensorrt']['device'] == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = DenseNet121(num_class= cfg['classes']).to(device)
    model_state_file = torch.load(cfg['saved_model_path'], map_location = device)['weight']
    model.load_state_dict(model_state_file)
    model.eval()
    return model 


def check_trt_model(cfg):
    trt_model_path = cfg['tensorrt']['trt_model_path']
    input_height = cfg['tensorrt']['height']
    input_width = cfg['tensorrt']['width']
    device = cfg['tensorrt']['device']

    try:
        if not os.path.exists(trt_model_path):
            net = build_tensorrt_model(cfg)
            print('ts file is not found. compiling a new ts file')

            traced_model = torch.jit.trace(net, torch.empty([1, 3, input_height, input_width]).to(device))
            trt_ts_module = torch_tensorrt.compile(traced_model, \
                            inputs = [torch.tensorrt.Input(
                            min_shape = [1, 3, input_height, input_width], \
                            opt_shape = [1, 3, input_height, input_width], \
                            max_shape = [1, 3, input_height, input_width], \
                            dtype = torch.half
                            )], \
                            require_full_compilation = True, enabled_precision = {torch.half}, truncate_long_and_double = True)
            torch.jit.save(trt_ts_module, trt_model_path)
            print('saved the .ts file')
        else:
            print('found trt file')
            trt_ts_module = torch.jit.load(trt_model_path, map_location = device)
        return trt_ts_module
    except Exception as e:
        print('Except in check trt model: ', e)


if __name__ == '__main__':

    with open('basic.yml', 'r') as ff:
        cfg = yaml.safe_load(ff)

    logger = set_logger('tensorrt.log')
    logger.info(pprint.pformat(cfg))

    trt_model = check_trt_model(cfg)

    image = torch.randn(1, 3, cfg['tensorrt']['width'], cfg['tensorrt']['height']).to(cfg['tensorrt']['device'])

    _, prediction = trt_model(image)
    print('prediction.shape: ', prediction.shape)

    out = torch.argmax(prediction, dim = 1) # 1, 100
    print('out: ', out)


    
