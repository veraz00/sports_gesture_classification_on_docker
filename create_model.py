import torchvision.models as models 
import torch.nn as nn 
import torch 
from torchsummary import summary 
import timm 

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


class SwinTransformer(nn.Module):
    def __init__(self, num_class = 2):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model.head = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(512, num_class)
        )
    
    def forward(self, x):
        out = torch.softmax(self.model(x), dim = 1)
        return out, out 


if __name__ == '__main__':
    model = SwinTransformer()
    sample = torch.randn(1, 3, 224, 224)
    _, out = model(sample)
    print('out.size: ', out.shape)
    summary(model, (3, 224, 224))