
import torchvision.models as models
import torch.nn as _nn


def get_resnet50_model():
    model = models.resnet50(weights= None, num_classes=10)
    print("ResNet-50 model created.")
    # TODO: check if we need to modify the first conv layer and maxpool for CIFAR10
    # since it's recommended to get better resutls.
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool = nn.Identity()
    return model
