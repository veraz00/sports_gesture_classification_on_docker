# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch
import json 
import albumentations as A
import pretrainedmodels
import argparse
import numpy as np
import torch.nn as nn

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


def validate_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # this will raise an exception if the file is not a valid image
        return True
    except:
        return False


def predict(image_path, model, device):
    test_datasets = Custom_dataset(
        image_list = [image_path], \
        label_list = [0], \
        transform=image_transforms['test']
    )
    # print('image_path', image_path)
    # print('len of test_dataset: ', len(test_datasets))

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    for img, _ in test_loader:
        img = img.to(device)
        _, out = model(img)  # out.shape = batchsize, num_classes
        predict_label = torch.argmax(out, dim = -1)  # batch, 1
        # print('predict result', predict_label.item(), label_list[predict_label.item()])
        return predict_label.item()


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
            
            

            pred_index = predict(image_location, model, device)  # label index
            pred_label = label_list[pred_index]
            logger.info(f'prediction result: {pred_index}, {pred_label}')

            return render_template("index.html", prediction=pred_label, label = pred_index, image_loc=image_file.filename)
    logger.info('Get request')
    return render_template("index.html", image_loc=None)


if __name__ == "__main__":
    logger = set_logger('flask.log')

    parser = argparse.ArgumentParser(description = 'flask for sport gesture api')
    parser.add_argument('--config', default = 'basic.yml', help = 'yaml file for training')

    args = parser.parse_args()
    logger.info('get args: %s', args)

    with open(args.config, 'r') as ff:
        cfg = yaml.safe_load(ff)

    label_list = list(json.load(open(cfg['saved_label_path'])).keys())
    logger.info(f'label_list: {" ".join(i for i in label_list)}')

    upload_folder = cfg['API']['upload_folder']
    device = cfg['API']['device']
    logger.info(f'upload_folder: {upload_folder}, device: {device}')


    model = DenseNet121(num_class= len(label_list))
    model_state = torch.load(cfg['saved_model_path'], map_location = torch.device(device))['weight']
    model.load_state_dict(model_state)
    model.to(device)
    logger.info('Loaded model!')

    app.run(host="0.0.0.0", port=12000, debug=True)
