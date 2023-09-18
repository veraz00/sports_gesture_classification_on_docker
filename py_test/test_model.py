import torch 
import pytest 
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')


from create_model import * 
from create_dataset import * 
from utils import * 

def test_dataloader():
    image_path = os.path.join(os.path.dirname(__file__), '../static/1.jpg')
    test_dataset = Custom_dataset(image_list = [image_path], label_list = [0], transform = image_transforms(224, 224, phase = 'test'))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = 1,
        shuffle = False,
        num_workers = 0)
    
    for img, _ in test_loader:
        assert img.shape == (1, 3, 224, 224)
        # assert img.any() <= 2.25 and img.any >= -2.12 
    return 

from icecream import ic 

def test_swin_transformer():
    fake_img = torch.randn((1, 3, 224, 224))
    model = SwinTransformer(num_class = 10)
    prediction, _ = model(fake_img, )
    ic(torch.sum(prediction, dim = -1))
    assert prediction.shape == (1, 10) 
    assert torch.sum(prediction, ) == 1
    return 
      

def test_densenet121():
    fake_img = torch.randn((1, 3, 224, 224))
    model = DenseNet121(num_class = 10)
    features, prediction = model(fake_img, )
    ic(torch.sum(prediction, dim = -1))

    assert prediction.shape == (1, 10) 
    assert abs(torch.sum(prediction, ).item() - 1.0) < 1e-6 
    assert features.shape == (1, 1024)
    return 
