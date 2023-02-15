import os
import torch

import torch.nn as nn
from torch.nn import functional as F

from torch import cuda
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ConfusionMatrix
import pretrainedmodels
import numpy as np
import pandas as pd

from apex import amp
from sklearn import metrics

from PIL import Image # only  get 
import cv2
import json
from tqdm import tqdm
# https://www.kaggle.com/datasets/gpiosenka/sports-classification/code

from create_model import *
from create_dataset import *

def train(args):
    fold = args.fold 
    pretrained_model = args.pretrained_model
    training_data_path = args.parent_dataset_dir 
    csv_filename = args.csv_filename

    df = pd.read_csv(os.path.join(training_data_path, csv_filename))

    # train_data_path = "/home/linlin/data"
    # model_path = './'
    # df = pd.read_csv(os.path.join(train_data_path, 'sports_with_fold.csv'))

    target_dict = {vv:i for i, vv in enumerate(df.labels.unique())}  # 100
    with open('label.json', 'w') as ff:
        json.dump(target_dict, ff)
    # print('len of target_dict: ', len(target_dict))
    df['labels_index'] = -1
    df['labels_index'] = df['labels'].apply(lambda x: target_dict[x])


    df = df[df['data set'] == 'train']
    # device = "cpu" 
    device = 'cuda'
    num_epochs = 50
    train_bs = 8
    valid_bs = 8

    best_auc_score = 0.6
    saved_path = 'Densenet121_sports_classification.pt'


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_images = df_train.filepaths.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.labels_index.values  # [11, 21, 30, ,,,]

    valid_images = df_valid.filepaths.values.tolist()
    valid_images = [os.path.join(training_data_path, i) for i in valid_images]
    valid_targets = df_valid.labels_index.values

    train_datasets = Custom_dataset(image_list = train_images, label_list = train_targets, transform = image_transforms['train'])
    valid_datasets = Custom_dataset(image_list = valid_images, label_list = valid_targets, transform = image_transforms['valid'])
    data_loaders = {'train': DataLoader(train_datasets, batch_size = train_bs, shuffle = True), \
                  'valid': DataLoader(valid_datasets, batch_size = valid_bs, shuffle = False)}
    trainimages, trainlabels = next(iter(data_loaders['train']))


    model = DenseNet121(num_class= len(target_dict)).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )
    best_auc_score = 0.0
    if pretrained_model:
        pretrained_model = torch.load(pretrained_model)
        print('pretrained_model.dict: ', pretrained_model.keys())
        model.load_state_dict(pretrained_model['weight'])
        optimizer.load_state_dict(pretrained_model['optimizer'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        best_auc_score = pretrained_model['epoch_auc_score']
        print(f'load pretrained model with {best_auc_score}')





    criterion = nn.CrossEntropyLoss(weight = None).cuda()
    

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:

            running_loss = 0
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            predict_result = []
            ground_truth = []
            # Iterate over data.
            with tqdm(total=len(data_loaders[phase]) * train_bs, \
                    desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
                for imgs, labels in data_loaders[phase]:
                    
                    if torch.sum(imgs != imgs) > 1:
                        print("Data contains Nan.")

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if torch.cuda.is_available():
                            imgs = imgs.to(device)
                            labels = labels.to(device, dtype = torch.long)
        
                        # Forward
                        _, out = model(imgs)  # out.shape = bs, 100
                        loss = criterion(out, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        predict_label = torch.argmax(out, dim = -1)
                        predict_result.append(predict_label.detach().cpu().numpy())
                        # ground_truth.append(F.one_hot(labels, num_classes = 100).detach().cpu().numpy())
                        ground_truth.append(labels.detach().cpu().numpy())
                        running_loss += loss.item()
                        pbar.update(imgs.shape[0])
                    
                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                predict_result = np.concatenate(predict_result)  # from 28, bachsize --> (28 * batchsize, )
                # print('predict_result.shape: ', predict_result.shape)
                ground_truth = np.concatenate(ground_truth)
                # print('groudth_shape: ', ground_truth.shape)

                confmat = ConfusionMatrix(task = 'multiclass', num_classes = len(target_dict))
                maxtric_result = confmat(torch.tensor(predict_result), torch.tensor(ground_truth))
                epoch_auc_score = (torch.trace(maxtric_result) / torch.sum(maxtric_result)).item()
                                
                # print(f"epoch={epoch}, stage = {phase}, auc={round(epoch_auc_score, 4)}")
                if epoch_auc_score > best_auc_score and phase == 'valid':
                    torch.save({
                                'epoch': epoch,
                                'weight': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'epoch_auc_score': epoch_auc_score
                                }, saved_path)
                    best_auc_score = epoch_auc_score
                else:
                    print(f'Keep the previous best model weight: {round(best_auc_score, 4)}')

                scheduler.step(epoch_auc_score)
                pbar.set_postfix(**{"stage": phase, "lr": optimizer.param_groups[0]['lr'], \
                                "(batch) loss": round(epoch_loss, 4), "epoch_auc_score":round(epoch_auc_score, 4)})

if __name__ == "__main__":
    # fake_image = torch.randn(16, 3, 224, 224)
    # model = DenseNet121(num_class= 100)
    # output, loss = model(fake_image,)
    # print('output_size: ', output.shape, output[0][0])
    import argparse
    parser = argparse.ArgumentParser('Args for training sport gesture detection')
    parser.add_argument('--parent_dataset_dir', type = str, default = '/home/linlin/dataset/sports_kaggle/')
    parser.add_argument('--csv_filename', type = str, help = 'csv file for dataset', default = 'sports_with_fold.csv')
    parser.add_argument('--fold', type = int, default = 0, help = 'fold usd for validation')
    
    
    parser.add_argument('--pretrained_model', type = str)
    
    args = parser.parse_args()

    train(args)
