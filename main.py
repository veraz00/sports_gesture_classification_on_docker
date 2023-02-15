import os
import torch
import torchvision.models as models 
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets 
from torch import cuda
from torch.utils.data import DataLoader, Dataset

import pretrainedmodels
import numpy as np
import pandas as pd

from apex import amp
from sklearn import metrics

from PIL import Image # only  get 
import cv2
import albumentations as A 
from tqdm import tqdm
# https://www.kaggle.com/datasets/gpiosenka/sports-classification/code

class DenseCrossEntropy(nn.Module):
    # Taken from:
    # https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()



# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# from torch.utils.data.dataloader import DataLoader


class DenseNet121(nn.Module):
    def __init__(self, num_class = 2):
        super(DenseNet121, self).__init__()
        dense_net = models.densenet121(pretrained = True)
        self.features = dense_net.features 
        self.fc = nn.Linear(in_features = dense_net.classifier.in_features, out_features = num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
    def forward(self, x):
        x = self.features(x)  # bs, 1024, 7, 7
        x = nn.ReLU(inplace = True)(x)
        x = self.avgpool(x).view(x.size(0), -1)  # bs, 1024
        
        out = torch.softmax(self.fc(x), dim = 1)
        return x, out 

class Custom_dataset(Dataset):
    def __init__(self, image_list, label_list, transform = None):
        self.image_list = image_list
        self.label_list = label_list 
        self.transform = transform 
        assert len(image_list) == len(label_list), "The number of images is not equal to the number of labels"
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index]) #if cv2: cannot use pytorch.transpose
        label = self.label_list[index]

        if self.transform is not None:
            # print('self.transform: ', self.transform)
            image = self.transform(image = image)['image']
        image=np.transpose(image,(2, 1, 0))
        return image, label

    def __len__(self):
        return len(self.image_list)

 

def train(fold = 0):
    training_data_path = "/home/linlin/dataset/sports_kaggle/"
    model_path = "./"
    df = pd.read_csv("/home/linlin/dataset/sports_kaggle/sports_with_fold.csv")

    # train_data_path = "/home/linlin/data"
    # model_path = './'
    # df = pd.read_csv(os.path.join(train_data_path, 'sports_with_fold.csv'))

    target_dict = {vv:i for i, vv in enumerate(df.labels.unique())}  # 100
    # print('len of target_dict: ', len(target_dict))
    df['labels_index'] = -1
    df['labels_index'] = df['labels'].apply(lambda x: target_dict[x])


    df = df[df['data set'] == 'train']
    device = "cpu" 
    num_epochs = 3
    train_bs = 8
    valid_bs = 8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    best_auc_score = 0.6
    saved_path = 'Densenet121_sports_classification.pt'


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    image_transforms = {'train': A.Compose([
    A.Resize(height = 224, width = 224, always_apply = True),
    A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
    A.Flip(p=0.5)
    ]), 
    'valid': A.Compose([
    A.Resize(height = 224, width = 224, always_apply = True),
    A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
    ])
    }


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

    criterion = nn.CrossEntropyLoss(weight = None).cuda()
    best_auc_score = 0.5
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
                        features, out = model(imgs)  # out.shape = bs, 100
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
                from torchmetrics import ConfusionMatrix
                confmat = ConfusionMatrix(task = 'multiclass', num_classes = len(target_dict))
                maxtric_result = confmat(torch.tensor(predict_result), torch.tensor(ground_truth))
                epoch_auc_score = torch.trace(maxtric_result) / torch.sum(maxtric_result)

                # confusion_matrix = metrics.confusion_matrix(ground_truth, predict_result, multi_class = 'ovo')
                                
                # print(f"epoch={epoch}, stage = {phase}, auc={round(epoch_auc_score, 4)}")
                if epoch_auc_score > best_auc_score:
                    torch.save({
                                'epoch': epoch,
                                'weight': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'loss': epoch_loss,
                                }, saved_path)
                    best_auc_score = epoch_auc_score

                scheduler.step(epoch_auc_score)
                pbar.set_postfix(**{"stage": phase, "lr": optimizer.param_groups[0]['lr'], "(batch) loss": epoch_loss, "epoch_auc_score":epoch_auc_score })

if __name__ == "__main__":
    fake_image = torch.randn(16, 3, 224, 224)
    model = DenseNet121(num_class= 100)
    output, loss = model(fake_image,)
    print('output_size: ', output.shape, output[0][0])
    train(fold=0)
