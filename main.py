import os
import logging 

import torch
import torch.nn as nn
from torch import cuda
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import ConfusionMatrix
from tensorboardX import SummaryWriter

import pretrainedmodels
import numpy as np
import pandas as pd
from PIL import Image # only  get 
import cv2
import json
from tqdm import tqdm
# https://www.kaggle.com/datasets/gpiosenka/sports-classification/code


from create_model import *
from create_dataset import *
from utils import *



def train(model, data_loaders, optimizer, scheduler, criterion, best_acc_score, cfg, logger):
    writer = SummaryWriter(cfg['tensorboard_name'] if cfg['tensorboard_name'] else 'mela_api')
    num_epochs = cfg['epochs']
    device = cfg['device']
    numpy_test = cfg['numpy_test']
    saved_model_path = cfg['saved_model_path']

    for epoch in range(num_epochs):
        # logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        logger.info('-' * 20)

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
            num_corrects, num_samples = 0, 0

            # Iterate over data.
            with tqdm(total=len(data_loaders[phase]) * data_loaders[phase].batch_size, \
                    desc=f'Epoch {epoch}/{num_epochs}', unit='img') as pbar:
                for imgs, labels in data_loaders[phase]:
                    
                    if torch.sum(imgs != imgs) > 1:
                        logger.info("Data contains Nan.")

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
                        if numpy_test:
                            predict_result.append(predict_label.detach().cpu().numpy())
                            # ground_truth.append(F.one_hot(labels, num_classes = 100).detach().cpu().numpy())
                            ground_truth.append(labels.detach().cpu().numpy())
                        else:
                            num_corrects += (predict_label == labels).sum().item()
                            num_samples += labels.size(0)
       
                        running_loss += loss.item()
                        pbar.update(imgs.shape[0])
                    
                if numpy_test:
                    predict_result = np.concatenate(predict_result)  # from 28, bachsize --> (28 * batchsize, )
                    # logger.info('predict_result.shape: ', predict_result.shape)
                    ground_truth = np.concatenate(ground_truth)
                    # logger.info('groudth_shape: ', ground_truth.shape)

                    confmat = ConfusionMatrix(task = 'multiclass', num_classes = int(cfg['classes']))
                    maxtric_result = confmat(torch.tensor(predict_result), torch.tensor(ground_truth))
                    epoch_acc_score = (torch.trace(maxtric_result) / torch.sum(maxtric_result)).item()
                else:
                    epoch_acc_score = num_corrects/num_samples

                epoch_loss = running_loss / len(data_loaders[phase].dataset)                
                # logger.info(f"epoch={epoch}, stage = {phase}, acc={round(epoch_acc_score, 4)}")
                if epoch_acc_score > best_acc_score and phase == 'valid':
                    torch.save({
                                'epoch': epoch,
                                'weight': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'epoch_acc_score': epoch_acc_score
                                }, saved_model_path)
                    best_acc_score = epoch_acc_score
                else:
                    logger.info(f'Keep the previous best model weight: {round(best_acc_score, 4)}')
                
                scheduler.step(epoch_acc_score) # epoch_loss
                pbar.set_postfix(**{"stage": phase, "lr": optimizer.param_groups[0]['lr'], \
                                "(batch) loss": round(epoch_loss, 4), "epoch_acc_score":round(epoch_acc_score, 4)})

            writer.add_scalar(f'{phase}_loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}_epoch_acc_score', epoch_acc_score, epoch)
            writer.add_scalar(f'{phase}_lr', optimizer.param_groups[0]['lr'], epoch)
            if phase == 'valid':
                writer.add_scalar(f'{phase}_best_acc_score', best_acc_score, epoch)



def generate_images_labels_from_csv(df, parent_data_dir, target_dict, label_column = 'labels', image_column = 'filepaths'):
    # generate absolute images path and label index 
          
    label_index_column = label_column + '_index'
    df[label_index_column] = -1
    df[label_index_column] = df[label_column].apply(lambda x: target_dict[x])
    images = df[image_column].values.tolist()
    if not images[0].startswith('/') and parent_data_dir.startswith('/'):
        images = [os.path.join(parent_data_dir, i) for i in images]
        targets = df[label_index_column].values  # [11, 21, 30, ,,,]
    return images, targets 


def main(args):
    logger = set_logger(file_name= 'record.log')
    logging.info("Input args: %r", args)
# The %s specifier converts the object using str(), and %r converts it using repr().
# but repr() is special that is valid Python syntax, which could be used to unambiguously \
# recreate the object it represents, like keeping \n.

    required_keys = ['config']
    for key in required_keys:
        if key not in args:
            raise ValueError(f'args is missing required key: {key}')

    with open(args.config, 'r') as ff:
        cfg = yaml.safe_load(ff)


    parent_data_dir = cfg['parent_dataset_dir'] 
    csv_file_path = cfg['csv_file_path']
    df  = pd.read_csv(csv_file_path)
    csv_column = {'images': cfg['columns']['images'], 'labels': cfg['columns']['labels'], 'split': cfg['columns']['split']}  # how to upgrade it 
    target_dict = {vv: i for i, vv in enumerate(df[csv_column['labels']].unique())}
    with open(cfg['saved_label_path'], 'w') as ff:
        json.dump(target_dict, ff)
    logger.info('len of target_dict: %s', len(target_dict))

    if csv_column['split'] == 'fold':
        df_train = df[df[csv_column['split']] != cfg['columns']['fold']].reset_index(drop=True)  # need to run create_fold to generate the fold 
        df_valid = df[df[csv_column['split']] == cfg['columns']['fold']].reset_index(drop=True)
    else:
        df_train = df[df[csv_column['split']] == 'train'].reset_index(drop=True)
        df_valid = df[df[csv_column['split']] == 'valid'].reset_index(drop=True)

    # train_data_path = "/home/linlin/data"
    # model_path = './'
    # df = pd.read_csv(os.path.join(train_data_path, 'sports_with_fold.csv'))


    train_images, train_targets = generate_images_labels_from_csv(df_train, parent_data_dir, target_dict, \
                                                    label_column=csv_column['labels'], \
                                                    image_column=csv_column['images'])

    valid_images, valid_targets = generate_images_labels_from_csv(df_valid, parent_data_dir, target_dict, \
                                                    label_column=csv_column['labels'], \
                                                    image_column=csv_column['images'])
    
    device = 'cuda' if torch.cuda.is_available() and cfg['device'] == 'cuda' else 'cpu'

    
    batch_sizes = {'train': cfg['train']['batch_size'], 'valid': cfg['valid']['batch_size']}



    train_datasets = Custom_dataset(image_list = train_images, label_list = train_targets, transform = image_transforms['train'])
    valid_datasets = Custom_dataset(image_list = valid_images, label_list = valid_targets, transform = image_transforms['valid'])
    data_loaders = {'train': DataLoader(train_datasets, batch_size = batch_sizes['train'], shuffle = True), \
                  'valid': DataLoader(valid_datasets, batch_size = batch_sizes['valid'], shuffle = False)}
    # trainimages, trainlabels = next(iter(data_loaders['train']))


    model = DenseNet121(num_class= len(target_dict)).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr = float(cfg['learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )  # Reduce learning rate when a metric has stopped improving. 
    criterion = nn.CrossEntropyLoss(weight = None).cuda()
    best_acc_score = 0.0

    if cfg['pretrained_model_path']:
        pretrained_model = torch.load(cfg['pretrained_model_path'])
        logger.info('pretrained_model.dict: ', pretrained_model.keys())
        model.load_state_dict(pretrained_model['weight'])
        optimizer.load_state_dict(pretrained_model['optimizer'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        best_acc_score = pretrained_model['epoch_acc_score']
        logger.info(f'load pretrained model with {best_acc_score}')

    

    train(model, data_loaders, optimizer, scheduler, criterion, best_acc_score, cfg, logger)

    logger.info('finish training!')


if __name__ == "__main__":
    # fake_image = torch.randn(16, 3, 224, 224)
    # model = DenseNet121(num_class= 100)
    # output, loss = model(fake_image,)
    # print('output_size: ', output.shape, output[0][0])

    import argparse
    import yaml 

    parser = argparse.ArgumentParser(description = 'sports gesture training')
    parser.add_argument('--config', default = 'basic.yml', help = 'yaml file for training')

    args = parser.parse_args()
    
    main(args)
