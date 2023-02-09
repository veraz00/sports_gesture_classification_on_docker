import os
import torch

import albumentations
import pretrainedmodels
import numpy as np
import pandas as pd
import torch.nn as nn

from apex import amp
from sklearn import metrics
from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationDataset
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 100)
    
    def forward(self, image, targets):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=100)
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)  # bs, 2048
        out = self.out(x)
        print('out.shape', out.shape)
        # print('targets_one_hot.shape', targets_one_hot.shape)
        loss = nn.BCEWithLogitsLoss()(
            out, targets_one_hot.type_as(out)
        )
        return out, loss


def train(fold):
    # training_data_path = "/home/linlin/dataset/sports_kaggle/"
    # model_path = "/home/linlin/ll_docker/melanoma-deep-learning"
    # df = pd.read_csv("/home/linlin/dataset/sports_kaggle/sports_with_fold.csv")

    train_data_path = "/home/linlin/data"
    model_path = './'
    df = pd.read_csv(os.path.join(train_data_path, 'sports_with_fold.csv'))

    target_dict = {vv:i for i, vv in enumerate(df.labels.unique())}  # 100
    print('len of target_dict: ', len(target_dict))
    df['labels_index'] = -1
    df['labels_index'] = df['labels'].apply(lambda x: target_dict[x])


    df = df[df['data set'] == 'train']
    device = "cuda"
    epochs = 3
    train_bs = 16
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.filepaths.values.tolist()
    train_images = [os.path.join(training_data_path, i) for i in train_images]
    train_targets = df_train.labels_index.values

    valid_images = df_valid.filepaths.values.tolist()
    valid_images = [os.path.join(training_data_path, i) for i in valid_images]
    valid_targets = df_valid.labels_index.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )
    # print('train_dataset', train_dataset[0])  # {images: tensor, targets: 72}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )


    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="O1",
        verbosity=0
    )
    Engine_device = Engine(model, optimizer, device, fp16 = False)

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        training_loss = Engine_device.train(
            train_loader,
            

            # fp16=True
        )
        predictions, valid_loss = Engine_device.evaluate(
            valid_loader
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break


if __name__ == "__main__":
    train(fold=0)
