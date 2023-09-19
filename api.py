# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch
import json 
import time 


import albumentations as A
import pretrainedmodels
import argparse
import pprint
import numpy as np
import torch.nn as nn
import onnx 
import onnxruntime as ort


from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F
from PIL import Image 
import yaml

from create_model import * 
from create_dataset import * 
from utils import * 


app = Flask(__name__)


logger = set_logger('flask.log')

# parser = argparse.ArgumentParser(description = 'flask for sport gesture api')
# parser.add_argument('--config', default = 'flask.yml', help = 'yaml file for training')

# args = parser.parse_args()
# logger.info('get args: %s', args)


# with open(args.config, 'r') as ff:
#     cfg = yaml.safe_load(ff)
# logger.info(pprint.pformat(args))
# logger.info(pprint.pformat(cfg))

labels = list(json.load(open('./label.json')).keys())
logger.info(f'labels: {" ".join(i for i in labels)}')

upload_folder = './static'
logger.info(f'upload_folder: {upload_folder}, ')

pt_path = './SwinTransformer_sports_classification.pt'
onnx_path = './SwinTransformer_sports_classification.onnx'


def get_onnx_model(model_name, pt_path, onnx_path):
    if model_name == 'SwinTransformer':
        model = SwinTransformer(num_class= len(labels))
        logger.info(f'use {model_name}')
        

    if not os.path.exists(onnx_path): 
        logger.info('not find onnx model. export it now')
        model_state = torch.load(pt_path)['weight']
        model.load_state_dict(model_state)
        torch.onnx.export(model, \
                        torch.randn(1, 3, 224, 224), \
                        onnx_path, \
                        export_params=True, \
                        opset_version=13, \
                        input_names = ['input'], \
                        output_names = ['features', 'output'])
        logger.info('exported onnx model')

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    providers= ['CPUExecutionProvider']
    model  = ort.InferenceSession(onnx_path, providers = providers)
    return model 


def validate_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # this will raise an exception if the file is not a valid image
        return True
    except:
        return False


def predict(image_path):
    test_datasets = Custom_dataset(
        image_list = [image_path], \
        label_list = [0], \
        transform=image_transforms(height = 224, width = 224, phase = 'test')
    )
    # print('image_path', image_path)
    # print('len of test_dataset: ', len(test_datasets))

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # check the time difference 
    # torch.cuda.synchronize()
    # start = time.time()

    for img, _ in test_loader:
        img = img

        model = get_onnx_model(model_name = 'SwinTransformer', pt_path = pt_path, onnx_path = onnx_path)
        input_name = model.get_inputs()[0].name
        # label_name = model.get_outputs()[0].name
        out = model.run(None, {input_name: img.cpu().numpy()})[-1]  # onnx model is for running on CPU -- its output is numpy 
        out = torch.tensor(out)
        # _, out = model(img)  # out.shape = batchsize, num_classes
        confidence, predict_label = torch.max(out, dim = -1)  # batch, 1
        # print('time: ', (time.time() - start))
        # print('predict result', predict_label.item(), labels[predict_label.item()])
        return predict_label.item(), confidence.item()


@app.route("/", methods=["GET", "POST"])
def upload_predict():   
    if request.method == "POST":
        if 'image' not in request.files:
            return "No image provided", 400
        
        image_file = request.files["image"]

        if image_file:
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            logger.info(f'saved_image: {image_location}')

            if validate_image(image_location) == False:
                raise ValueError('Invalid image file')
            else:
                logger.info('passed the image check')

            pred_index, confidence = predict(image_location)  # label index
            pred_label = labels[pred_index]
            logger.info(f'prediction result: {pred_index}, {pred_label}, confidence: {confidence}')

            return render_template("index.html", prediction=pred_label, label = pred_index, confidence = confidence, image_loc=image_file.filename)
    logger.info('Get request')
    return render_template("index.html", image_loc=None)


if __name__ == "__main__":
        

    app.run(host="0.0.0.0", port=12000, debug=False)
