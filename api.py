# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch

import albumentations as A
import pretrainedmodels

import numpy as np
import torch.nn as nn

from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F
import json 

from create_model import * 
from create_dataset import * 

app = Flask(__name__)
UPLOAD_FOLDER = "/home/linlin/ll_docker/melanoma-deep-learning/static"
DEVICE = "cuda"
model_path = 'Densenet121_sports_classification.pt'


label_path = 'label.json'
label_list = list(json.load(open(label_path)).keys())
# print('label_list: ', label_list)



def predict(image_path, model):
    test_datasets = Custom_dataset(
        image_list = [image_path], \
        label_list = [0], \
        transform=image_transforms['test']
    )
    print('len of test_dataset: ', len(test_datasets))

    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    for img, label in test_loader:
        img = img.to(DEVICE)
        _, out = model(img)  # out.shape = batchsize, num_classes
        predict_label = torch.argmax(out, dim = -1)  # batch, 1
        print('predict result', predict_label.item(), label_list[predict_label.item()])
        return predict_label.item()


@app.route("/", methods=["GET", "POST"])
def upload_predict():   
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred_index = predict(image_location, MODEL)  # label index
            pred_label = label_list[pred_index]
            # print('prediction result: ', pred_index, pred_label)
            return render_template("index.html", prediction=pred_label, label = pred_index, image_loc=image_file.filename)
    return render_template("index.html", image_loc=None)


if __name__ == "__main__":
    MODEL = DenseNet121(num_class= len(label_list))
    model_state = torch.load(model_path, map_location = torch.device(DEVICE))['weight']
    MODEL.load_state_dict(model_state)
    MODEL.to(DEVICE)
    # app.run(host="0.0.0.0", port=12000, debug=True)
    predict(image_path = '/home/linlin/ll_docker/melanoma-deep-learning/static/002.jpg', model = MODEL)
