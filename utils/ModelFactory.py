import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
import os

class ModelFactory():

    @staticmethod
    def get_network(name, target_classes):
        
        set_seed()
        #https://docs.pytorch.org/vision/main/models/alexnet.html
        if name == 'alexnet':
            retargetted_model = torchvision.models.alexnet(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/convnext.html
        elif name == 'convnext_tiny':
            retargetted_model = torchvision.models.convnext_tiny(weights="DEFAULT")
            retargetted_model.classifier[2] = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'convnext_small':
            retargetted_model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights)
            retargetted_model.classifier[2] = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'convnext_base':
            retargetted_model = torchvision.models.convnext_base(weights="DEFAULT")
            retargetted_model.classifier[2] = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'convnext_large':
            retargetted_model = torchvision.models.convnext_large(weights="DEFAULT")
            retargetted_model.classifier[2] = nn.Linear(in_features=1536, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/densenet.html
        elif name == 'densenet121':
            retargetted_model = torchvision.models.densenet121(weights="DEFAULT")
            retargetted_model.classifier = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'densenet169':
            retargetted_model = torchvision.models.densenet169(weights="DEFAULT")
            retargetted_model.classifier = nn.Linear(in_features=1664, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'densenet161':
            retargetted_model = torchvision.models.densenet161(weights="DEFAULT")
            retargetted_model.classifier = nn.Linear(in_features=2208, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'densenet201':
            retargetted_model = torchvision.models.densenet201(weights="DEFAULT")
            retargetted_model.classifier = nn.Linear(in_features=1920, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/efficientnetv2.html
        elif name == 'efficientnetV2_s':
            retargetted_model = torchvision.models.efficientnet_v2_s(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'efficientnetV2_m':
            retargetted_model = torchvision.models.efficientnet_v2_m(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'efficientnetV2_l':
            retargetted_model = torchvision.models.efficientnet_v2_l(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/googlenet.html
        elif name == "googlenet":
            retargetted_model = torchvision.models.googlenet(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/inception.html
        elif name == "inception_v3":
            retargetted_model = torchvision.models.inception_v3(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model  
        elif name == "inception_v3_untrained":
            retargetted_model = torchvision.models.get_model("inception_v3", weights=None)
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/mnasnet.html
        elif name == "mnasnet0_5":
            retargetted_model = torchvision.models.mnasnet0_5(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "mnasnet0_75":
            retargetted_model = torchvision.models.mnasnet0_75(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "mnasnet1_0":
            retargetted_model = torchvision.models.mnasnet1_0(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "mnasnet1_3":
            retargetted_model = torchvision.models.mnasnet1_3(weights="DEFAULT")
            retargetted_model.classifier[1] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/maxvit.html
        elif name == "maxVit":
            retargetted_model = torchvision.models.maxvit_t( weights="DEFAULT")
            retargetted_model.classifier[5] = nn.Linear(in_features=512, out_features=target_classes, bias=False)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/mobilenetv3.html
        elif name == "mobilenet_large":
            retargetted_model = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
            retargetted_model.classifier[3] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "mobilenet_small":
            retargetted_model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
            retargetted_model.classifier[3] = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/resnet.html
        elif name == 'resnet18':
            retargetted_model = torchvision.models.resnet18(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=512, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'resnet34':
            retargetted_model = torchvision.models.resnet34(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=512, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'resnet50':
            retargetted_model = torchvision.models.resnet50(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'resnet101':
            retargetted_model = torchvision.models.resnet101(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'resnet152':
            retargetted_model = torchvision.models.resnet152(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/resnext.html
        elif name == "resnext50_32x4d":
            retargetted_model = torchvision.models.resnext50_32x4d(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "resnext101_32x8d":
            retargetted_model = torchvision.models.resnext50_32x4d(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "resnext101_64x4d":
            retargetted_model = torchvision.models.resnext50_32x4d(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/shufflenetv2.html
        elif name == "shufflenet_x0_5":
            retargetted_model = torchvision.models.shufflenet_v2_x0_5(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "shufflenet_x1":
            retargetted_model = torchvision.models.shufflenet_v2_x1_0(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "shufflenet_x1_5":
            retargetted_model = torchvision.models.shufflenet_v2_x1_5(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "shufflenet_x2":
            retargetted_model = torchvision.models.shufflenet_v2_x2_0(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        #TODO: Implement SqeezeNet
        # elif name == 'squeezenet1_0':
        #     retargetted_model = torchvision.models.resnet152(pretrained=True)
        #     retargetted_model.classifier[1] = nn.Conv2d(in_features=512, out_features=target_classes, kernel_size=(1, 1), stride=(1, 1))
        #https://docs.pytorch.org/vision/main/models/swin_transformer.html
        elif name == "swin_v2_t":
            retargetted_model = torchvision.models.swin_v2_t(weights="DEFAULT")
            retargetted_model.head = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "swin_v2_s":
            retargetted_model = torchvision.models.swin_v2_s(weights="DEFAULT")
            retargetted_model.head = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "swin_v2_b":
            retargetted_model = torchvision.models.swin_v2_b(weights="DEFAULT")
            retargetted_model.head = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/vision_transformer.html
        elif name == "vit_b_16":
            retargetted_model = torchvision.models.vit_b_16(weights="DEFAULT")
            retargetted_model.heads[0] = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "vit_b_32":
            retargetted_model = torchvision.models.vit_b_32(weights="DEFAULT")
            retargetted_model.heads[0] = nn.Linear(in_features=768, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "vit_l_16":
            retargetted_model = torchvision.models.vit_l_16(weights="DEFAULT")
            retargetted_model.heads[0] = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "vit_l_32":
            retargetted_model = torchvision.models.vit_l_32(weights="DEFAULT")
            retargetted_model.heads[0] = nn.Linear(in_features=1024, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "vit_h_32":
            retargetted_model = torchvision.models.vit_h_14(weights="DEFAULT")
            retargetted_model.heads[0] = nn.Linear(in_features=1280, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/vgg.html
        elif name == 'VGG11':
            retargetted_model = torchvision.models.vgg11(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG11_bn':
            retargetted_model = torchvision.models.vgg11_bn(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG13':
            retargetted_model = torchvision.models.vgg13(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG13_bn':
            retargetted_model = torchvision.models.vgg13_bn(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG16':
            retargetted_model = torchvision.models.vgg16(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG16_bn':
            retargetted_model = torchvision.models.vgg16_bn(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG19':
            retargetted_model = torchvision.models.vgg19(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == 'VGG19_bn':
            retargetted_model = torchvision.models.vgg19_bn(weights="DEFAULT")
            retargetted_model.classifier[6] = nn.Linear(in_features=4096, out_features=target_classes, bias=True)
            return retargetted_model
        #https://docs.pytorch.org/vision/main/models/wide_resnet.html
        elif name == "wide_resnet50":
            retargetted_model = torchvision.models.wide_resnet50_2(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "wide_resnet101":
            retargetted_model = torchvision.models.wide_resnet101_2(weights="DEFAULT")
            retargetted_model.fc = nn.Linear(in_features=2048, out_features=target_classes, bias=True)
            return retargetted_model
        elif name == "custom":
            #you could also return a custom built network
            return nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(6),#added in a batch norm
                nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=12*4*4, out_features=120),
                nn.ReLU(),
                nn.BatchNorm1d(120), #added in a batch norm
                nn.Linear(in_features=120, out_features=60),
                nn.ReLU(),
                nn.Linear(in_features=60, out_features=target_classes)
            )
        else:
            return None
        
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
