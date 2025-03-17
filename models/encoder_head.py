import torch.nn as nn
from models.resnetcifar import ResNet50_cifar10, ResNet18_cifar10
from models.cnncifar import CNNCifar_header
from models.cnnmnist import CNNMnist_header
from models.cnnhar import CNNHAR_header
from models.squeezenet import SqueezeNet_header
from models.mobilenetV1 import Mobilenet_v1
from models.lstm import ModelLSTMShakespeare
from models.simple_cnn import SimpleCNN_header
from models.mobileformer import MobileFormer
import torch.nn.functional as F
import torch



cfg = {
        'name': 'mf151',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 12,
        # stage1
        'bneck': {'e': 24, 'o': 12, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 16, 'exp': 48, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 16, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 32, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 32, 'exp': 192, 'out': 64, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 64, 'exp': 256, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 64, 'exp': 384, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 88, 'exp': 528, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 88, 'exp': 528, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1280,  # hid_layer
        'fc2': 58   # num_classes`
    }


def feature_extract(base_model):
    if base_model == "resnet18-cifar10" or base_model == "resnet18":
        basemodel = ResNet18_cifar10()
        # basemodel = resnet18(0, projector=False, model_rate=1, track=False)
        features = nn.Sequential(*list(basemodel.children())[:-2])
        num_ftrs = basemodel.fc.in_features
    elif base_model == 'cnn-cifar':
        features = CNNCifar_header()
        # features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        num_ftrs = 512
    elif base_model == 'cnn-mnist':
        features = CNNMnist_header()
        num_ftrs = 50
    elif base_model == 'cnn-har':
        features = CNNHAR_header()
        num_ftrs = 128
    elif base_model == 'squeezenet-cifar':
        features = SqueezeNet_header()
        num_ftrs = 8192
    elif base_model == 'mobilenet-tsrd':
        features = Mobilenet_v1(1)
        num_ftrs = 1024
    elif base_model == 'lstm-shakespeare':
        features = ModelLSTMShakespeare()
        num_ftrs = 256
    elif base_model == 'mobileformer-tsrd':
        features = MobileFormer(cfg)
        num_ftrs = 1280
    elif base_model == 'resnet50-imagenet':
        basemodel = ResNet50_cifar10()
        features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
    else:
        exit("Error:no models (encoder_head.py)")
    return features, num_ftrs

class ModelFedCon(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        h = h.squeeze()
        z = self.l1(h)
        z = F.relu(z)
        z = self.l2(z)
        y = self.l3(z)
        return h, z, y

class ModelFedCon_cnn(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_cnn, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        z = self.l1(h)
        z = F.relu(z)
        z = self.l2(z)
        y = self.l3(z)
        return h, h, y
        
        
        
class ModelFedCon_squeezenet(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_squeezenet, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(512, 10, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 512x4x4 -> 10x1x1
            )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h, y = self.features(x)
        #h = torch.flatten(x, 1)
        #h = h.squeeze()
        #x = self.classifier(x)  # torch.Size([1, 10, 1, 1])
        #y = x.view(x.size(0), -1)  # torch.Size([1, 10])


        return h, h, y
        
        
class ModelFedCon_mobilenet(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_mobilenet, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):
        h, y = self.features(x)
        #h = torch.flatten(x, 1)
        #h = h.squeeze()
        #x = self.classifier(x)  # torch.Size([1, 10, 1, 1])
        #y = x.view(x.size(0), -1)  # torch.Size([1, 10])


        return h, h, y
        

class ModelFedCon_lstm(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_lstm, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)

    def forward(self, x):
        h = self.features(x)
        #h = torch.flatten(x, 1)
        #h = h.squeeze()
        #x = self.classifier(x)  # torch.Size([1, 10, 1, 1])
        #y = x.view(x.size(0), -1)  # torch.Size([1, 10])


        return h, h, h

class ModelFedCon_mobileformer(nn.Module):
    def __init__(self, base_model, out_dim, n_classes):
        super(ModelFedCon_mobileformer, self).__init__()
        self.features, num_ftrs = feature_extract(base_model=base_model)

    def forward(self, x):
        h = self.features(x)
        return h, h, h



class ModelFedCon_noheader(nn.Module):

    def __init__(self, base_model, n_classes):
        super(ModelFedCon_noheader, self).__init__()

        self.features, num_ftrs = feature_extract(base_model=base_model)
        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        # print("h size:", h.size())
        y = self.l3(h)
        return h, h, y

class ModelFedCon_cnn_noheader(nn.Module):

    def __init__(self, base_model, n_classes):
        super(ModelFedCon_cnn_noheader, self).__init__()

        self.features, num_ftrs = feature_extract(base_model=base_model)
        # last layer
        self.l3 = nn.Linear(num_ftrs, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze()
        # print("h size:", h.size())
        y = self.l3(h)
        return h, h, y





