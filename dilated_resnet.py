"""
copied from irn
modified by zhengsz@pku.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50


class Net(nn.Module):

    def __init__(self, num_classes=200):
        super(Net, self).__init__()

        self.resnet50 = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

        self.d1 = nn.Conv2d(2048, 2048, 3, dilation=1, bias=False, padding=1)
        # self.d2 = nn.Conv2d(2048, 512, 3, dilation=2, bias=False, padding=2)
        # self.d3 = nn.Conv2d(2048, 512, 3, dilation=3, bias=False, padding=3)
        # self.d4 = nn.Conv2d(2048, 512, 3, dilation=4, bias=False, padding=4)
        # self.d6 = nn.Conv2d(2048, 512, 3, dilation=6, bias=False, padding=6)
        # self.d8 = nn.Conv2d(2048, 512, 3, dilation=8, bias=False, padding=8)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.d1])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.d1(x)
        # x2 = self.d2(x)
        # x3 = self.d3(x)
        # x4 = self.d4(x)
        # x6 = self.d6(x)
        # x8 = self.d8(x)

        # x = x1 + (x2 + x3 + x4 + x6 + x8) / 5

        x = torch.relu(x)

        x = torch.mean(x.view(x.size(0), x.size(1), -1), -1).unsqueeze(-1).unsqueeze(-1)

        x = self.classifier(x).squeeze()

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet50.layer1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self, num_classes=200):
        super(CAM, self).__init__(num_classes)

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x1 = F.conv2d(x, self.d1.weight, bias=None, stride=1, padding=1, dilation=1)
        x2 = F.conv2d(x, self.d1.weight, bias=None, stride=1, padding=2, dilation=2)
        x3 = F.conv2d(x, self.d1.weight, bias=None, stride=1, padding=3, dilation=3)
        x6 = F.conv2d(x, self.d1.weight, bias=None, stride=1, padding=6, dilation=6)
        x9 = F.conv2d(x, self.d1.weight, bias=None, stride=1, padding=9, dilation=9)

        x = x1 + (x2 + x3 + x6 + x9) / 4

        x = F.conv2d(x, self.classifier.weight)

        return x
